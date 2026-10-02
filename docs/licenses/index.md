# Native JPEG notices

Vision builds statically link the checksum-pinned libjpeg-turbo 3.2.0 codec through turbojpeg 1.5.1's safe Rust API. Include these notices when distributing a vision-enabled binary:

- [libjpeg-turbo license](libjpeg-turbo-LICENSE.md): upstream license overview and BSD notice.
- [Independent JPEG Group notice](README.ijg): IJG license and required acknowledgement. This software is based in part on the work of the Independent JPEG Group.
- [Rust turbojpeg binding license](turbojpeg-LICENSE): MIT license for the safe binding.

Source: [official libjpeg-turbo 3.2.0 release](https://github.com/libjpeg-turbo/libjpeg-turbo/releases/tag/3.2.0), archive SHA-256 `6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e`; [Rust binding source](https://github.com/honzasp/rust-turbojpeg). Model Apache-2.0 notices are separate and preserved among the 14 cached model files.
