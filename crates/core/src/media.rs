//! Bounded still-image decoding and Qwen3.5 patch preparation.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "image dimensions and interpolation coordinates are bounded before arithmetic"
)]
use std::{
    fmt::{self, Debug, Formatter},
    io::Cursor,
};

use bytes::Bytes;
use image::{ImageFormat, ImageReader, Limits, RgbImage};
use turbojpeg::{Decompressor, Image as JpegImage, PixelFormat};

use crate::{Error, Result};

/// Validated RGB pixels; debug output never prints image data.
#[derive(Clone)]
pub struct Image {
    pixels: Bytes,
    width: u32,
    height: u32,
}
impl Debug for Image {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}
impl Image {
    /// Accept RGB8 pixels with explicit dimensions and bounded storage.
    ///
    /// # Errors
    /// Rejects invalid dimensions, byte count or pixel limits.
    pub fn from_rgb(width: u32, height: u32, pixels: Bytes) -> Result<Self> {
        if width == 0
            || height == 0
            || width > 4096
            || height > 4096
            || u64::from(width) * u64::from(height) > 4_000_000
            || u64::from(width.max(height)) > u64::from(width.min(height)) * 200
        {
            return Err(Error::LimitExceeded("image dimensions".into()));
        }
        let count = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| Error::LimitExceeded("pixel bytes".into()))?;
        if usize::try_from(count).ok() != Some(pixels.len()) {
            return Err(Error::InvalidRequest("RGB pixel count".into()));
        }
        Ok(Self {
            pixels,
            width,
            height,
        })
    }
    /// Decode only static PNG/JPEG, with matching MIME and probe-first limits.
    ///
    /// # Errors
    /// Rejects format mismatch, animation, corrupt input or excessive allocations.
    pub fn decode(media_type: &str, data: &Bytes) -> Result<Self> {
        if data.is_empty() || data.len() > 2 * 1024 * 1024 {
            return Err(Error::LimitExceeded("encoded image bytes".into()));
        }
        let format = match media_type {
            "image/png" => ImageFormat::Png,
            "image/jpeg" => ImageFormat::Jpeg,
            _ => return Err(Error::UnsupportedCapability("image format".into())),
        };
        if image::guess_format(data).ok() != Some(format) {
            return Err(Error::InvalidRequest("image MIME/format mismatch".into()));
        }
        if format == ImageFormat::Png {
            png_metadata(data)?;
        } else {
            jpeg_metadata(data)?;
        }
        if format == ImageFormat::Jpeg {
            return decode_jpeg(data);
        }
        let (width, height) = ImageReader::with_format(Cursor::new(data), format)
            .into_dimensions()
            .map_err(|_| Error::InvalidRequest("image dimensions".into()))?;
        if width == 0
            || height == 0
            || width > 4096
            || height > 4096
            || u64::from(width) * u64::from(height) > 4_000_000
        {
            return Err(Error::LimitExceeded("decoded image dimensions".into()));
        }
        let mut reader = ImageReader::with_format(Cursor::new(data), format);
        let mut limits = Limits::default();
        limits.max_image_width = Some(4096);
        limits.max_image_height = Some(4096);
        limits.max_alloc = Some(64 * 1024 * 1024);
        reader.limits(limits);
        let decoded = reader
            .decode()
            .map_err(|_| Error::InvalidRequest("image decode failed".into()))?
            .to_rgb8();
        Self::from_rgb(width, height, Bytes::from(decoded.into_raw()))
    }
    /// Image dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub(crate) fn pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }
    pub(crate) fn reservation_bytes(&self) -> Result<usize> {
        let (height, width) = smart_resize(self.height as usize, self.width as usize)?;
        height
            .checked_mul(width)
            .and_then(|n| n.checked_mul(32))
            .and_then(|n| n.checked_add(self.pixels.len()))
            .ok_or_else(|| Error::LimitExceeded("prepared image bytes".into()))
    }
    pub(crate) fn prepare(&self) -> Result<PreparedImage> {
        let (height, width) = smart_resize(self.height as usize, self.width as usize)?;
        let resized = resize_rgb(
            &self.pixels,
            self.width as usize,
            self.height as usize,
            width,
            height,
        )?;
        let gh = height / 16;
        let gw = width / 16;
        let count = gh * gw;
        if count / 4 > 4096 {
            return Err(Error::LimitExceeded("visual token count".into()));
        }
        // [block_h, block_w, merge_h, merge_w, channel, temporal, patch_h, patch_w]
        let mut patches = Vec::with_capacity(count * 3 * 2 * 16 * 16);
        let mut coordinates = Vec::with_capacity(count);
        for bh in 0..gh / 2 {
            for bw in 0..gw / 2 {
                for mh in 0..2 {
                    for mw in 0..2 {
                        let ph = bh * 2 + mh;
                        let pw = bw * 2 + mw;
                        coordinates.push([ph, pw]);
                        for channel in 0..3 {
                            for _ in 0..2 {
                                for y in 0..16 {
                                    for x in 0..16 {
                                        let pixel = resized
                                            .get_pixel((pw * 16 + x) as u32, (ph * 16 + y) as u32)
                                            .0
                                            .get(channel)
                                            .copied()
                                            .ok_or_else(|| {
                                                Error::InvalidRequest("RGB channel".into())
                                            })?;
                                        patches.push((f32::from(pixel) / 255. - 0.5) / 0.5);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(PreparedImage {
            patches,
            grid: [1, gh, gw],
            coordinates,
        })
    }
}
#[derive(Clone)]
pub(crate) struct PreparedImage {
    pub patches: Vec<f32>,
    pub grid: [usize; 3],
    pub coordinates: Vec<[usize; 2]>,
}
impl Debug for PreparedImage {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedImage")
            .field("grid", &self.grid)
            .finish_non_exhaustive()
    }
}
pub(crate) fn smart_resize(height: usize, width: usize) -> Result<(usize, usize)> {
    let factor = 32_usize;
    let mut h = ((height as f64 / factor as f64).round_ties_even() as usize) * factor;
    let mut w = ((width as f64 / factor as f64).round_ties_even() as usize) * factor;
    // Enforce operator visual-token ceiling independently of upstream maximum.
    if h * w > 4_194_304 {
        return Err(Error::LimitExceeded("resized pixels".into()));
    }
    if h * w < 65536 {
        let beta = (65536. / (height * width) as f64).sqrt();
        h = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        w = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    if h == 0 || w == 0 || h * w > 4_194_304 {
        return Err(Error::LimitExceeded("resized image".into()));
    }
    Ok((h, w))
}
fn png_metadata(bytes: &[u8]) -> Result<()> {
    let mut offset = 8_usize;
    let mut metadata = 0_usize;
    if bytes.get(24) == Some(&16) {
        return Err(Error::UnsupportedCapability("16-bit PNG".into()));
    }
    while offset < bytes.len() {
        let size: u32 = u32::from_be_bytes(
            bytes
                .get(offset..offset + 4)
                .ok_or_else(|| Error::InvalidRequest("PNG chunk".into()))?
                .try_into()
                .map_err(|_| Error::InvalidRequest("PNG chunk size".into()))?,
        );
        let kind = bytes
            .get(offset + 4..offset + 8)
            .ok_or_else(|| Error::InvalidRequest("PNG chunk kind".into()))?;
        if matches!(kind, b"iCCP" | b"zTXt" | b"iTXt") {
            return Err(Error::UnsupportedCapability(
                "compressed or extended PNG metadata".into(),
            ));
        }
        if kind == b"acTL" {
            return Err(Error::UnsupportedCapability("animated PNG".into()));
        }
        if kind != b"IDAT" {
            metadata = metadata
                .checked_add(size as usize)
                .ok_or_else(|| Error::LimitExceeded("PNG metadata".into()))?;
            if metadata > 65536 {
                return Err(Error::LimitExceeded("PNG metadata".into()));
            }
        }
        offset = offset
            .checked_add(size as usize)
            .and_then(|n| n.checked_add(12))
            .ok_or_else(|| Error::InvalidRequest("PNG chunk overflow".into()))?;
        if offset > bytes.len() {
            return Err(Error::InvalidRequest("PNG truncated chunk".into()));
        }
        if kind == b"IEND" {
            if offset != bytes.len() {
                return Err(Error::InvalidRequest("PNG trailing bytes".into()));
            }
            return Ok(());
        }
    }
    Err(Error::InvalidRequest("PNG missing end".into()))
}

// This is the only JPEG FFI boundary. The dependency exposes a safe owned API;
// all native buffers are bounded, owned, and synchronous, and never cross actors.
fn decode_jpeg(data: &Bytes) -> Result<Image> {
    let failure = |_| Error::InvalidRequest("JPEG decode failed".into());
    let mut decoder = Decompressor::new().map_err(failure)?;
    decoder.set_scan_limit(32).map_err(failure)?;
    let header = decoder.read_header(data).map_err(failure)?;
    let width =
        u32::try_from(header.width).map_err(|_| Error::LimitExceeded("JPEG width".into()))?;
    let height =
        u32::try_from(header.height).map_err(|_| Error::LimitExceeded("JPEG height".into()))?;
    if width == 0
        || height == 0
        || width > 4096
        || height > 4096
        || u64::from(width) * u64::from(height) > 4_000_000
    {
        return Err(Error::LimitExceeded("JPEG dimensions".into()));
    }
    let pitch = header
        .width
        .checked_mul(3)
        .ok_or_else(|| Error::LimitExceeded("JPEG row".into()))?;
    let length = pitch
        .checked_mul(header.height)
        .ok_or_else(|| Error::LimitExceeded("JPEG pixels".into()))?;
    let mut pixels = vec![0; length];
    decoder
        .decompress(
            data,
            JpegImage {
                pixels: &mut pixels,
                width: header.width,
                height: header.height,
                pitch,
                format: PixelFormat::RGB,
            },
        )
        .map_err(failure)?;
    Image::from_rgb(width, height, Bytes::from(pixels))
}

fn jpeg_metadata(bytes: &[u8]) -> Result<()> {
    let mut offset = 2_usize;
    let mut metadata = 0_usize;
    while offset < bytes.len() {
        if bytes.get(offset) != Some(&0xff) {
            return Err(Error::InvalidRequest("JPEG marker".into()));
        }
        while bytes.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let marker = *bytes
            .get(offset)
            .ok_or_else(|| Error::InvalidRequest("JPEG marker".into()))?;
        offset += 1;
        if matches!(marker, 0xc0..=0xc2)
            && (bytes.get(offset.saturating_add(2)) != Some(&8)
                || !matches!(bytes.get(offset.saturating_add(7)), Some(1 | 3)))
        {
            return Err(Error::UnsupportedCapability(
                "JPEG precision/components".into(),
            ));
        }
        if marker == 0xda || marker == 0xd9 {
            return Ok(());
        }
        if marker == 0x01 || (0xd0..=0xd8).contains(&marker) {
            continue;
        }
        let prefix: [u8; 2] = bytes
            .get(offset..offset.saturating_add(2))
            .ok_or_else(|| Error::InvalidRequest("JPEG segment".into()))?
            .try_into()
            .map_err(|_| Error::InvalidRequest("JPEG segment length".into()))?;
        let size = usize::from(u16::from_be_bytes(prefix));
        if size < 2 {
            return Err(Error::InvalidRequest("JPEG segment length".into()));
        }
        let end = offset
            .checked_add(size)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| Error::InvalidRequest("JPEG truncated segment".into()))?;
        if (0xe0..=0xef).contains(&marker) || marker == 0xfe {
            metadata = metadata
                .checked_add(size)
                .ok_or_else(|| Error::LimitExceeded("JPEG metadata".into()))?;
            if metadata > 65536 {
                return Err(Error::LimitExceeded("JPEG metadata".into()));
            }
            if marker == 0xe2
                && bytes.get(offset.saturating_add(2)..offset.saturating_add(6)) == Some(b"MPF\0")
            {
                return Err(Error::UnsupportedCapability("multi-picture JPEG".into()));
            }
        }
        offset = end;
    }
    Err(Error::InvalidRequest("JPEG missing image data".into()))
}

// Match Torch uint8 antialiased bicubic: round and clip after each axis.
fn resize_rgb(input: &[u8], iw: usize, ih: usize, ow: usize, oh: usize) -> Result<RgbImage> {
    fn coefficients(input: usize, output: usize) -> (Vec<Vec<(usize, i64)>>, u32) {
        let scale = input as f64 / output as f64;
        let filter_scale = scale.max(1.);
        let support = 2. * filter_scale;
        let rows: Vec<Vec<(usize, f64)>> = (0..output)
            .map(|i| {
                let center = (i as f64 + 0.5) * scale;
                let start = ((center - support + 0.5).floor().max(0.) as usize).min(input);
                let end = ((center + support + 0.5).floor().max(0.) as usize).min(input);
                let mut weights = Vec::new();
                let mut sum = 0.;
                for position in start..end {
                    let x = ((position as f64 - center + 0.5) / filter_scale).abs();
                    let weight = if x < 1. {
                        ((1.5 * x - 2.5) * x) * x + 1.
                    } else if x < 2. {
                        (((-0.5 * x + 2.5) * x - 4.) * x) + 2.
                    } else {
                        0.
                    };
                    sum += weight;
                    weights.push((position, weight));
                }
                weights.into_iter().map(|(i, w)| (i, w / sum)).collect()
            })
            .collect();
        let maximum = rows.iter().flatten().map(|(_, w)| *w).fold(0_f64, f64::max);
        let mut precision = 0_u32;
        while precision < 22 {
            if (0.5 + maximum * f64::from(1_u32 << (precision + 1))) as u32 >= 32768 {
                break;
            }
            precision += 1;
        }
        let scale = f64::from(1_u32 << precision);
        (
            rows.into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|(i, w)| (i, (w * scale).round() as i64))
                        .collect()
                })
                .collect(),
            precision,
        )
    }
    let (horizontal, hprecision) = coefficients(iw, ow);
    let (vertical, vprecision) = coefficients(ih, oh);
    let mut intermediate = vec![0_u8; ow * ih * 3];
    let mut output = vec![0_u8; ow * oh * 3];
    for y in 0..ih {
        for (x, weights) in horizontal.iter().enumerate() {
            for channel in 0..3 {
                let mut sum = 1_i64 << (hprecision - 1);
                for (position, weight) in weights {
                    sum += i64::from(
                        *input
                            .get((y * iw + position) * 3 + channel)
                            .ok_or_else(|| Error::InvalidRequest("resize RGB offset".into()))?,
                    ) * weight;
                }
                let pixel = ((sum >> hprecision).clamp(0, 255)) as u8;
                *intermediate
                    .get_mut((y * ow + x) * 3 + channel)
                    .ok_or_else(|| Error::InvalidRequest("resize horizontal offset".into()))? =
                    pixel;
            }
        }
    }
    for (y, weights) in vertical.iter().enumerate() {
        for x in 0..ow {
            for channel in 0..3 {
                let mut sum = 1_i64 << (vprecision - 1);
                for (position, weight) in weights {
                    sum += i64::from(
                        *intermediate
                            .get((position * ow + x) * 3 + channel)
                            .ok_or_else(|| {
                                Error::InvalidRequest("resize vertical source".into())
                            })?,
                    ) * weight;
                }
                *output
                    .get_mut((y * ow + x) * 3 + channel)
                    .ok_or_else(|| Error::InvalidRequest("resize vertical offset".into()))? =
                    ((sum >> vprecision).clamp(0, 255)) as u8;
            }
        }
    }
    RgbImage::from_raw(ow as u32, oh as u32, output)
        .ok_or_else(|| Error::InvalidRequest("resized RGB shape".into()))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    #[test]
    fn test_should_bound_pixels_and_duplicate_temporal_patches() -> Result<()> {
        assert!(Image::from_rgb(0, 32, Bytes::new()).is_err());
        assert!(Image::from_rgb(4096, 4096, Bytes::new()).is_err());
        let image = Image::from_rgb(256, 256, Bytes::from(vec![255; 256 * 256 * 3]))?;
        let prepared = image.prepare()?;
        assert_eq!(prepared.grid, [1, 16, 16]);
        assert_eq!(prepared.patches.len(), 256 * 1536);
        assert!(
            prepared
                .patches
                .iter()
                .all(|x| (*x - 1.).abs() < f32::EPSILON)
        );
        Ok(())
    }
    #[test]
    fn test_should_bound_jpeg_decoder_rounding_and_metadata() -> Result<()> {
        let data = Bytes::from_static(include_bytes!("../fixtures/synthetic/image-37-61.jpg"));
        let image = Image::decode("image/jpeg", &data)?;
        let expected = include_bytes!("../fixtures/synthetic/image-37-61-jpeg.rgb");
        let errors: Vec<_> = image
            .pixels
            .iter()
            .zip(expected)
            .map(|(a, b)| a.abs_diff(*b))
            .collect();
        assert_eq!(image.pixels.len(), expected.len());
        assert_eq!(errors.iter().copied().max(), Some(0));
        assert!(Image::decode("image/png", &data).is_err());
        let mut metadata = vec![0xff, 0xd8];
        for _ in 0..2 {
            metadata.extend([0xff, 0xe1, 0xff, 0xff]);
            metadata.extend(vec![0; 65533]);
        }
        assert!(matches!(
            jpeg_metadata(&metadata),
            Err(Error::LimitExceeded(_))
        ));
        assert!(jpeg_metadata(&[0xff, 0xd8, 0xff, 0xe1, 0, 1]).is_err());
        let image = Image::from_rgb(256, 256, Bytes::from(vec![255; 256 * 256 * 3]))?;
        assert!(image.reservation_bytes()? >= image.prepare()?.patches.len() * 4);
        Ok(())
    }
    proptest! {
        #![proptest_config(ProptestConfig { cases: 10_000, max_shrink_iters: 1024, ..ProptestConfig::default() })]
        #[test]
        #[ignore = "10,000-case native-boundary mutation fuzz campaign; run make fuzz-media"]
        fn test_should_fuzz_bounded_image_decoders(mutations in prop::collection::vec((0_usize..8192, any::<u8>()), 0..32)) {
            let mut data = include_bytes!("../fixtures/synthetic/image-37-61.jpg").to_vec();
            for (offset, byte) in mutations {
                let position = offset % data.len();
                if let Some(slot) = data.get_mut(position) { *slot = byte; }
            }
            if let Ok(image) = Image::decode("image/jpeg",&Bytes::from(data)) {
                let (width,height) = image.dimensions();
                prop_assert!(width <= 4096 && height <= 4096);
                prop_assert!(u64::from(width)*u64::from(height) <= 4_000_000);
                prop_assert_eq!(image.pixels.len(), width as usize * height as usize * 3);
            }
        }
    }
}
