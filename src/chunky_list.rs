use std::cell::UnsafeCell;

use smallvec::SmallVec;

struct TempListInner<T> {
    payload: SmallVec<[T; 8]>,
}

impl<T> Default for TempListInner<T> {
    fn default() -> Self {
        Self {
            payload: Default::default(),
        }
    }
}

pub struct TempList<T>(UnsafeCell<Box<TempListInner<T>>>);

impl<T> Default for TempList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TempList<T> {
    pub fn new() -> Self {
        Self(UnsafeCell::new(Box::default()))
    }

    pub fn add(&self, item: T) -> &T {
        unsafe {
            let inner = &mut *self.0.get();
            inner.payload.push(item);

            &inner.payload[inner.payload.len() - 1]
        }
    }
}
