mod manager;
mod manager_lock;

pub use manager::{UnsafeResourceHandle, UnsafeResourceManager};
pub use manager_lock::{ResourceHandle, ResourceHooks, ResourceId, ResourceManager};
