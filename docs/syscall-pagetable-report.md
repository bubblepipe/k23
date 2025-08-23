# System Call and Page Table Implementation Report

## 1. Function Call as System Call

### Current State

The k23 kernel currently **does not have a traditional system call mechanism**. All operations run in kernel mode with direct function calls.

#### Evidence from the Codebase

Looking at the shell implementation (`kernel/src/shell.rs`), commands are implemented as direct kernel functions:

```rust
// kernel/src/shell.rs:302-310
const SHUTDOWN: Command = Command::new("shutdown")
    .with_help("exit the kernel and shutdown the machine.")
    .with_fn(|_| {
        tracing::info!("Bye, Bye!");

        global().executor.stop();

        Ok(())
    });
```

These are not system calls but direct function invocations within the kernel itself.

### How System Calls Would Work

#### Traditional System Call Flow

1. **User space** makes a function call that triggers a trap/exception
2. **CPU switches** from user mode to kernel mode  
3. **Trap handler** identifies the system call and dispatches
4. **Kernel** performs the requested operation with kernel privileges
5. **Return** to user space with result

#### RISC-V System Call Implementation

For RISC-V, system calls would use the `ecall` instruction:

```rust
// Example of what would be needed in kernel/src/arch/riscv64/trap_handler.rs
// (Currently this only handles page faults and timer interrupts)

match scause::read().cause() {
    Trap::Exception(Exception::UserEnvCall) => {
        // System call from user mode
        let syscall_num = a7;  // RISC-V convention
        let args = [a0, a1, a2, a3, a4, a5];
        
        match syscall_num {
            SYSCALL_READ => sys_read(args),
            SYSCALL_WRITE => sys_write(args),
            SYSCALL_EXIT => sys_exit(args[0]),
            // ...
        }
    }
    // ... other trap cases
}
```

Current RISC-V trap handler (`kernel/src/arch/riscv64/trap_handler.rs`):

```rust
// kernel/src/arch/riscv64/trap_handler.rs:50-70
extern "C" fn supervisor_trap_handler() {
    let scause = scause::read();
    let sepc = sepc::read();
    let stval = stval::read();

    match scause.cause() {
        Trap::Exception(Exception::StorePageFault)
        | Trap::Exception(Exception::LoadPageFault)
        | Trap::Exception(Exception::InstructionPageFault) => {
            let virt = VirtualAddress::new(stval).unwrap();
            let flags = match scause.cause() {
                Trap::Exception(Exception::StorePageFault) => PageFaultFlags::STORE,
                Trap::Exception(Exception::LoadPageFault) => PageFaultFlags::LOAD,
                Trap::Exception(Exception::InstructionPageFault) => PageFaultFlags::INSTRUCTION,
                _ => unreachable!(),
            };

            handle_page_fault(virt, flags).expect("failed to handle page fault");
        }
        // ... timer and IPI handling
```

#### x86_64 System Call Implementation

For x86_64, modern systems use the `syscall` instruction (or legacy `int 0x80`):

```asm
; User space system call
mov rax, SYSCALL_NUMBER  ; System call number
mov rdi, arg1            ; First argument
mov rsi, arg2            ; Second argument
syscall                  ; Trigger system call
```

The kernel would need MSR setup:

```rust
// What would be needed in kernel/src/arch/x86_64/mod.rs
unsafe fn setup_syscall_msrs() {
    // STAR MSR: Segment selectors for syscall/sysret
    wrmsr(0xC0000081, kernel_cs << 32 | user_cs << 48);
    
    // LSTAR MSR: Syscall entry point
    wrmsr(0xC0000082, syscall_entry as u64);
    
    // SYSCALL_MASK MSR: RFLAGS mask
    wrmsr(0xC0000084, 0x47700);  // Mask interrupts, etc.
}
```

Currently, x86_64 trap handling is not implemented (`kernel/src/arch/x86_64/trap_handler.rs`):

```rust
// kernel/src/arch/x86_64/trap_handler.rs:13-19
/// Initialize the trap handler for this CPU.
pub fn init() {
    // TODO: Initialize x86_64 interrupt descriptor table (IDT)
    // TODO: Set up exception handlers for page faults, general protection faults, etc.
    // For now, we'll continue without trap handling - this means any exception will triple fault
    tracing::warn!("x86_64 trap handler not yet implemented - exceptions will cause triple fault");
}
```

### Required Infrastructure for System Calls

1. **User/Kernel Mode Separation**: Currently everything runs in kernel mode
2. **System Call Table**: Mapping of syscall numbers to handlers
3. **ABI Definition**: Register usage conventions for arguments/returns
4. **Context Switching**: Save/restore user state
5. **Security Checks**: Validate user pointers, permissions

## 2. Page Table Implementation

### Overview

The k23 kernel implements a sophisticated page table system with architecture-specific backends for x86_64 and RISC-V.

### Loader Page Table Setup

#### x86_64 Boot Page Tables

The x86_64 loader creates initial page tables during the 32-bit to 64-bit transition:

```rust
// loader/src/arch/x86_64.rs:34-42
pub const DEFAULT_ASID: u16 = 0;
pub const KERNEL_ASPACE_BASE: usize = 0xffffffc000000000;
pub const PAGE_SIZE: usize = 4096;
pub const PAGE_TABLE_ENTRIES: usize = 512;
pub const PAGE_TABLE_LEVELS: usize = 4; // PML4, PDPT, PD, PT
pub const VIRT_ADDR_BITS: u32 = 48;

pub const PAGE_SHIFT: usize = (PAGE_SIZE - 1).count_ones() as usize;
pub const PAGE_ENTRY_SHIFT: usize = (PAGE_TABLE_ENTRIES - 1).count_ones() as usize;
```

Boot assembly creates minimal identity mapping:

```asm
// loader/src/arch/x86_64.rs:67-90 (assembly in naked_asm!)
// Clear page table area
"mov edi, 0x1000",
"xor eax, eax",
"mov ecx, 0x1000",   // Clear 16KB (4096 dwords = 16KB)
"rep stosd",

// Set up PML4[0] -> PDPT (for identity mapping low addresses)
"mov dword ptr [0x1000], 0x2003",  // PDPT address | Present | Writable

// Set up PDPT[0] -> PD (for identity mapping)
"mov dword ptr [0x2000], 0x3003",  // PD address | Present | Writable

// Set up PD entries for first 256MB (128 * 2MB pages)
// Using 2MB pages (bit 7 = PS)
"mov edi, 0x3000",
"mov eax, 0x83",     // Present | Writable | PS (2MB pages)
"mov ecx, 128",      // 128 entries for 256MB
"2:",
"mov [edi], eax",
"add eax, 0x200000", // Next 2MB
"add edi, 8",
"loop 2b",
```

#### Page Table Entry Structure

```rust
// loader/src/arch/x86_64.rs:430-434
#[repr(transparent)]
pub struct PageTableEntry {
    bits: usize,
}

// loader/src/arch/x86_64.rs:127-141
bitflags! {
    #[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
    pub struct PTEFlags: usize {
        const VALID = 1 << 0;      // Present bit
        const WRITABLE = 1 << 1;   // Write permission
        const USER = 1 << 2;       // User accessible
        const PWT = 1 << 3;        // Write through
        const PCD = 1 << 4;        // Cache disable
        const ACCESSED = 1 << 5;   // Accessed bit
        const DIRTY = 1 << 6;      // Dirty bit
        const HUGE = 1 << 7;       // Page size bit (PS)
        const GLOBAL = 1 << 8;     // Global page
        const NX = 1 << 63;        // No execute
    }
}
```

#### Mapping Functions

The loader provides mapping functions for setting up kernel virtual memory:

```rust
// loader/src/arch/x86_64.rs:279-351
pub unsafe fn map_contiguous(
    root_pgtable: usize,
    frame_alloc: &mut FrameAllocator,
    mut virt: usize,
    mut phys: usize,
    len: NonZero<usize>,
    flags: Flags,
    phys_off: usize,
) -> crate::Result<()> {
    let mut remaining_bytes = len.get();

    'outer: while remaining_bytes > 0 {
        let mut pgtable: NonNull<PageTableEntry> = pgtable_ptr_from_phys(root_pgtable, phys_off);

        for lvl in (0..PAGE_TABLE_LEVELS).rev() {
            let index = pte_index_for_level(virt, lvl);
            let pte = unsafe { pgtable.add(index).as_mut() };

            if !pte.is_valid() {
                if can_map_at_level(virt, phys, effective_remaining, lvl) {
                    let page_size = page_size_for_level(lvl);
                    let mut pte_flags = PTEFlags::VALID | PTEFlags::from(flags);

                    // For large pages (2MB at level 1, 1GB at level 2), set the PS bit
                    if lvl > 0 {
                        pte_flags |= PTEFlags::HUGE;
                    }

                    pte.replace_address_and_flags(phys, pte_flags);
                    // ... advance pointers
                }
                // ... handle non-leaf entries
            }
        }
    }
}
```

### Kernel Page Table Management

#### Generic Interface

The kernel defines a generic interface for architecture-specific implementations:

```rust
// kernel/src/mem/mod.rs:240-280
pub trait ArchAddressSpace {
    type Flags: From<Permissions> + bitflags::Flags;

    fn new(asid: u16, frame_alloc: &FrameAllocator) -> crate::Result<(Self, Flush)>
    where
        Self: Sized;
    
    fn from_active(asid: u16) -> (Self, Flush)
    where
        Self: Sized;

    unsafe fn map_contiguous(
        &mut self,
        frame_alloc: &FrameAllocator,
        virt: VirtualAddress,
        phys: PhysicalAddress,
        len: NonZeroUsize,
        flags: Self::Flags,
        flush: &mut Flush,
    ) -> crate::Result<()>;

    unsafe fn update_flags(
        &mut self,
        virt: VirtualAddress,
        len: NonZeroUsize,
        new_flags: Self::Flags,
        flush: &mut Flush,
    ) -> crate::Result<()>;

    unsafe fn unmap(
        &mut self,
        virt: VirtualAddress,
        len: NonZeroUsize,
        flush: &mut Flush,
    ) -> crate::Result<()>;

    unsafe fn query(&mut self, virt: VirtualAddress) -> Option<(PhysicalAddress, Self::Flags)>;

    unsafe fn activate(&self);

    fn new_flush(&self) -> Flush;
}
```

#### x86_64 Kernel Implementation

```rust
// kernel/src/arch/x86_64/mem.rs:42-48
/// The portion of the virtual address space reserved for the kernel.
pub const KERNEL_ASPACE_RANGE: RangeInclusive<VirtualAddress> = RangeInclusive {
    start: VirtualAddress::new(0xffffffc000000000).unwrap(),
    end: VirtualAddress::MAX,
};

/// The portion of the virtual address space reserved for userspace.
pub const USER_ASPACE_RANGE: RangeInclusive<VirtualAddress> = RangeInclusive {
    start: VirtualAddress::MIN,
    end: VirtualAddress::new(0x00007fffffffffff).unwrap(),
};
```

Reading current page table from CR3:

```rust
// kernel/src/arch/x86_64/mem.rs:163-174
fn from_active(asid: u16) -> (Self, Flush) {
    // Read the current CR3 value
    let cr3_val: u64;
    unsafe {
        core::arch::asm!("mov {}, cr3", out(reg) cr3_val);
    }
    
    // Extract the page table physical address (bits 12-51)
    let root_pgtable = PhysicalAddress::new((cr3_val & !0xFFF) as usize);
    
    // ... create AddressSpace structure
}
```

TLB invalidation:

```rust
// kernel/src/arch/x86_64/mem.rs:76-87
pub fn invalidate_range(range: Range<VirtualAddress>) {
    // For now, just invalidate each page individually
    // TODO: Consider using invlpcg for larger ranges or full TLB flush
    let start_page = range.start.align_down(PAGE_SIZE);
    let end_page = range.end.checked_align_up(PAGE_SIZE).unwrap();
    
    let mut current = start_page;
    while current < end_page {
        unsafe {
            core::arch::asm!("invlpg [{}]", in(reg) current.get());
        }
        current = current.checked_add(PAGE_SIZE).unwrap();
    }
}
```

#### RISC-V Kernel Implementation

```rust
// kernel/src/arch/riscv64/mem.rs:32-38
/// The portion of the virtual address space reserved for the kernel.
pub const KERNEL_ASPACE_RANGE: RangeInclusive<VirtualAddress> = RangeInclusive {
    start: VirtualAddress::new(0xffffffc000000000).unwrap(),
    end: VirtualAddress::MAX,
};

/// The portion of the virtual address space reserved for userspace (Sv39).
pub const USER_ASPACE_RANGE: RangeInclusive<VirtualAddress> = RangeInclusive {
    start: VirtualAddress::MIN,
    end: VirtualAddress::new(0x0000003fffffffff).unwrap(),
};
```

### Page Fault Handling

The kernel implements demand paging through page fault handlers:

```rust
// kernel/src/mem/trap_handler.rs
pub fn handle_page_fault(virt: VirtualAddress, flags: PageFaultFlags) -> crate::Result<()> {
    // 1. Find the address space region containing the fault address
    // 2. Check if access is allowed based on flags
    // 3. Allocate physical frame if needed (demand paging)
    // 4. Map the page
    // 5. Return to retry the faulting instruction
}
```

### Virtual Memory Features

#### 1. Address Space Management

High-level address space operations (`kernel/src/mem/address_space.rs`):
- Region tracking and management
- Permission enforcement
- Lazy allocation support

#### 2. Virtual Memory Objects (VMOs)

```rust
// kernel/src/mem/vmo.rs
pub struct Vmo {
    size: usize,
    pages: Vec<Option<PhysicalFrame>>,
    // Copy-on-write, shared memory support
}
```

#### 3. Memory Mapped Files

```rust
// kernel/src/mem/mmap.rs
pub struct Mmap {
    aspace: Arc<Mutex<AddressSpace>>,
    range: Range<VirtualAddress>,
    // File backing, permissions, etc.
}
```

### Page Size Support

Both architectures support multiple page sizes:

| Architecture | Level | Page Size | Usage |
|-------------|-------|-----------|-------|
| **x86_64** | | | |
| | PT (Level 0) | 4 KB | Default pages |
| | PD (Level 1) | 2 MB | Large pages (PS bit) |
| | PDPT (Level 2) | 1 GB | Huge pages (PS bit) |
| **RISC-V** | | | |
| | Level 0 | 4 KB | Default pages |
| | Level 1 | 2 MB | Megapages |
| | Level 2 | 1 GB | Gigapages |

### Current Limitations

#### x86_64 Specific
- Incomplete `map_contiguous`, `unmap`, `query` implementations in kernel
- No PCID (Process Context ID) support for TLB optimization
- IDT and exception handling not implemented

#### General Limitations
- No swapping/paging to disk
- No shared memory between processes (no process abstraction yet)
- No copy-on-write optimization
- No fork() system call (would need COW)
- Single address space (kernel only, no user processes)

### Future Work Required

1. **Complete x86_64 implementation**
   - Finish page table manipulation functions
   - Implement IDT and exception handlers
   - Add PCID support for better TLB performance

2. **User space support**
   - Process abstraction with separate address spaces
   - System call interface
   - User/kernel memory isolation

3. **Advanced features**
   - Copy-on-write pages
   - Shared memory regions
   - Memory-mapped files
   - Swap/paging to disk

## Conclusion

The k23 kernel has a solid foundation for virtual memory with sophisticated page table management, but lacks the system call interface needed for true user/kernel separation. The page table implementation is production-ready for kernel operations but would need enhancement for multi-process support with proper isolation.