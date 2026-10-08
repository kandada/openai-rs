// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Automatic retry with exponential backoff.
//!
//! Used by both sync and async clients. Mirrors the retry logic in official
//! OpenAI Python SDK — retries on 429, 5xx, and network errors with jitter.

use std::thread;
use std::time::Duration;

/// Configuration for automatic retries.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (default: 3).
    pub max_retries: u32,
    /// Base delay between retries in milliseconds (default: 1000).
    pub base_delay_ms: u64,
    /// Maximum delay cap in milliseconds (default: 60000).
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_retries: 3,
            base_delay_ms: 1000,
            max_delay_ms: 60000,
        }
    }
}

impl RetryConfig {
    pub fn new(max_retries: u32) -> Self {
        RetryConfig {
            max_retries,
            ..Default::default()
        }
    }

    /// Compute delay for attempt `n` (0-indexed) with jitter.
    ///
    /// Delay grows exponentially as `base_delay_ms * 2^attempt` up to
    /// `max_delay_ms`, then a ±25% jitter is applied. Uses saturating
    /// arithmetic so large attempt counts can never overflow.
    pub fn delay_ms(&self, attempt: u32) -> u64 {
        // Cap the exponent so the shift is always well-defined.
        let exp = attempt.min(30);
        let base = self.base_delay_ms.saturating_mul(1u64 << exp);
        let capped = base.min(self.max_delay_ms);
        // Add jitter: ±25%
        let jitter = (capped as f64 * 0.25 * (rand_f64() * 2.0 - 1.0)) as u64;
        capped.saturating_add(jitter)
    }

    /// Sleep for the delay (sync version).
    pub fn sleep(&self, attempt: u32) {
        let ms = self.delay_ms(attempt);
        thread::sleep(Duration::from_millis(ms));
    }

    /// Sleep for a caller-supplied delay (e.g. from a `Retry-After` header).
    pub fn sleep_for(&self, delay: Duration) {
        thread::sleep(delay);
    }

    /// Return a future that sleeps (async version).
    #[cfg(feature = "async")]
    pub async fn async_sleep(&self, attempt: u32) {
        let ms = self.delay_ms(attempt);
        tokio::time::sleep(Duration::from_millis(ms)).await;
    }

    /// Return a future that sleeps for a caller-supplied delay (async).
    #[cfg(feature = "async")]
    pub async fn async_sleep_for(&self, delay: Duration) {
        tokio::time::sleep(delay).await;
    }
}

/// Deterministic-ish random value in [0, 1).
///
/// `RandomState::new()` seeds a fresh hasher from the OS each call, so the
/// result is effectively random without pulling in a `rand` dependency.
/// A monotonic counter is folded in so consecutive calls within a process
/// never produce identical streams.
fn rand_f64() -> f64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut h = RandomState::new().build_hasher();
    h.write_u64(n);
    (h.finish() as f64) / (u64::MAX as f64)
}

/// Execute a sync closure with retry logic, returning the result or last error.
pub fn retry_sync<F, T, E>(
    mut f: F,
    config: &RetryConfig,
    is_retryable: fn(&E) -> bool,
) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
{
    let mut last_err: Option<E> = None;
    for attempt in 0..=config.max_retries {
        match f() {
            Ok(val) => return Ok(val),
            Err(e) => {
                if attempt == config.max_retries || !is_retryable(&e) {
                    return Err(e);
                }
                last_err = Some(e);
                config.sleep(attempt);
            }
        }
    }
    Err(last_err.unwrap())
}

/// Execute an async closure with retry logic.
#[cfg(feature = "async")]
pub async fn retry_async<F, Fut, T, E>(
    mut f: F,
    config: &RetryConfig,
    is_retryable: fn(&E) -> bool,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
{
    let mut last_err: Option<E> = None;
    for attempt in 0..=config.max_retries {
        match f().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                if attempt == config.max_retries || !is_retryable(&e) {
                    return Err(e);
                }
                last_err = Some(e);
                config.async_sleep(attempt).await;
            }
        }
    }
    Err(last_err.unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_grows_exponentially() {
        let config = RetryConfig::default();
        let d0 = config.delay_ms(0);
        let d2 = config.delay_ms(2);
        assert!(d2 > d0, "delay should grow: {d0} -> {d2}");
    }

    #[test]
    fn test_delay_capped() {
        let config = RetryConfig {
            max_retries: 10,
            base_delay_ms: 1000,
            max_delay_ms: 5000,
        };
        for i in 5..10 {
            assert!(config.delay_ms(i) <= 5000 + 1250, "delay should be capped");
        }
    }

    #[test]
    fn test_delay_never_overflows_for_huge_attempt() {
        let config = RetryConfig::default();
        // u32::MAX attempts must not panic or overflow.
        let d = config.delay_ms(u32::MAX);
        assert!(d <= config.max_delay_ms * 2);
    }

    #[test]
    fn test_retry_succeeds_after_failures() {
        let config = RetryConfig {
            max_retries: 3,
            base_delay_ms: 1,
            max_delay_ms: 10,
        };
        let mut calls = 0;
        let result: Result<i32, &str> = retry_sync(
            || {
                calls += 1;
                if calls < 3 {
                    Err("fail")
                } else {
                    Ok(42)
                }
            },
            &config,
            |_| true,
        );
        assert_eq!(result.unwrap(), 42);
        assert_eq!(calls, 3);
    }

    #[test]
    fn test_retry_stops_on_non_retryable() {
        let config = RetryConfig {
            max_retries: 3,
            base_delay_ms: 1,
            max_delay_ms: 10,
        };
        let mut calls = 0;
        let result: Result<i32, &str> = retry_sync(
            || {
                calls += 1;
                Err("fatal")
            },
            &config,
            |_| false, // never retryable
        );
        assert!(result.is_err());
        assert_eq!(calls, 1);
    }
}
