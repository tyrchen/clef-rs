//! Bounded open sockets with an idle read deadline, including slow HTTP headers.
use std::{
    io,
    net::SocketAddr,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use axum::{
    extract::connect_info::Connected,
    serve::{IncomingStream, Listener},
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::{OwnedSemaphorePermit, Semaphore},
    time::{Instant, Sleep, sleep},
};

#[derive(Debug)]
pub(crate) struct LimitedListener {
    pub listener: TcpListener,
    pub capacity: Arc<Semaphore>,
    pub idle: Duration,
}
#[derive(Debug)]
pub(crate) struct Socket {
    stream: TcpStream,
    _permit: OwnedSemaphorePermit,
    timer: Pin<Box<Sleep>>,
    idle: Duration,
    active: Arc<AtomicUsize>,
}

#[derive(Debug, Clone)]
pub(crate) struct Connection(Arc<AtomicUsize>);
impl Connected<IncomingStream<'_, LimitedListener>> for Connection {
    fn connect_info(stream: IncomingStream<'_, LimitedListener>) -> Self {
        Self(stream.io().active.clone())
    }
}
#[derive(Debug)]
pub(crate) struct ActiveRequest(Arc<AtomicUsize>);
impl Connection {
    pub fn active(&self) -> ActiveRequest {
        self.0.fetch_add(1, Ordering::AcqRel);
        ActiveRequest(self.0.clone())
    }
}
impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
impl Listener for LimitedListener {
    type Io = Socket;
    type Addr = SocketAddr;
    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let Ok(permit) = self.capacity.clone().acquire_owned().await else {
                sleep(Duration::from_millis(100)).await;
                continue;
            };
            match self.listener.accept().await {
                Ok((stream, addr)) => {
                    return (
                        Socket {
                            stream,
                            _permit: permit,
                            timer: Box::pin(sleep(self.idle)),
                            idle: self.idle,
                            active: Arc::new(AtomicUsize::new(0)),
                        },
                        addr,
                    );
                }
                Err(_) => sleep(Duration::from_millis(100)).await,
            }
        }
    }
    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}
impl AsyncRead for Socket {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if this.active.load(Ordering::Acquire) > 0 {
            this.timer.as_mut().reset(Instant::now() + this.idle);
        }
        if this.timer.as_mut().poll(cx).is_ready() {
            return Poll::Ready(Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "socket idle timeout",
            )));
        }
        let before = buf.filled().len();
        match Pin::new(&mut this.stream).poll_read(cx, buf) {
            Poll::Ready(Ok(())) => {
                if buf.filled().len() > before {
                    this.timer.as_mut().reset(Instant::now() + this.idle);
                }
                Poll::Ready(Ok(()))
            }
            other => other,
        }
    }
}
impl AsyncWrite for Socket {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        this.timer.as_mut().reset(Instant::now() + this.idle);
        Pin::new(&mut this.stream).poll_write(cx, buf)
    }
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().stream).poll_flush(cx)
    }
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().stream).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use std::future::IntoFuture;

    use axum::{Router, extract::ConnectInfo, routing::get};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        sync::oneshot,
    };

    use super::*;

    async fn slow(ConnectInfo(connection): ConnectInfo<Connection>) -> &'static str {
        let _active = connection.active();
        sleep(Duration::from_millis(120)).await;
        "completed"
    }

    #[tokio::test]
    async fn test_should_expire_slow_headers_without_interrupting_active_requests()
    -> anyhow::Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let limited = LimitedListener {
            listener,
            capacity: Arc::new(Semaphore::new(2)),
            idle: Duration::from_millis(40),
        };
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(
            axum::serve(
                limited,
                Router::new()
                    .route("/slow", get(slow))
                    .into_make_service_with_connect_info::<Connection>(),
            )
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .into_future(),
        );
        let mut stream = TcpStream::connect(address).await?;
        stream
            .write_all(b"GET /slow HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .await?;
        let mut bytes = Vec::new();
        tokio::time::timeout(
            Duration::from_secs(2),
            stream.take(4096).read_to_end(&mut bytes),
        )
        .await??;
        let response = String::from_utf8(bytes)?;
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.ends_with("completed"));
        let mut stream = TcpStream::connect(address).await?;
        stream.write_all(b"GET /slow ").await?;
        let mut bytes = Vec::new();
        tokio::time::timeout(
            Duration::from_secs(2),
            stream.take(4096).read_to_end(&mut bytes),
        )
        .await??;
        assert_eq!(bytes.len(), 0);
        let _ = stop.send(());
        task.await??;
        Ok(())
    }
}
