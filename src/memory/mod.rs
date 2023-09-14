mod manager_lock;
mod manager;

pub use manager_lock::{ResourceManager, ResourceHooks, ResourceHandle};
pub use manager::{UnsafeResourceManager, UnsafeResourceHandle};