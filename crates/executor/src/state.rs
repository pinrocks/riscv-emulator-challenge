use std::{
    fs::File,
    io::{Seek, Write},
};

use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use std::sync::Arc;
use std::collections::VecDeque;
use crate::{
    events::MemoryRecord,
    syscalls::{SyscallCode, Syscall},
    ExecutorMode,
    Instruction,
    Opcode,
};

const SCHEDULER_WINDOW_SIZE: usize = 8;

#[derive(Debug, Clone)]
struct InstructionScheduler {
    window: VecDeque<Instruction>,
    dependencies: Vec<Vec<usize>>, // Track register dependencies
}

impl Default for InstructionScheduler {
    fn default() -> Self {
        Self {
            window: VecDeque::with_capacity(SCHEDULER_WINDOW_SIZE),
            dependencies: vec![Vec::new(); 32], // One vec per register
        }
    }
}

impl InstructionScheduler {
    fn add_instruction(&mut self, inst: Instruction) -> Option<Instruction> {
        if self.window.len() >= SCHEDULER_WINDOW_SIZE {
            // Window is full, return the best instruction to execute
            self.get_ready_instruction()
        } else {
            self.window.push_back(inst);
            None
        }
    }

    fn get_ready_instruction(&mut self) -> Option<Instruction> {
        // Find instruction with no dependencies
        for i in 0..self.window.len() {
            let inst = &self.window[i];
            if !self.has_dependencies(inst) {
                let inst = self.window.remove(i).unwrap();
                self.update_dependencies(&inst);
                return Some(inst);
            }
        }
        None
    }

    fn has_dependencies(&self, inst: &Instruction) -> bool {
        // Check register dependencies
        match inst.opcode {
            Opcode::ADD | Opcode::SUB | Opcode::XOR | Opcode::OR | Opcode::AND |
            Opcode::SLL | Opcode::SRL | Opcode::SRA | Opcode::MUL => {
                let (_, rs1, rs2) = inst.r_type();
                !self.dependencies[rs1 as usize].is_empty() || 
                !self.dependencies[rs2 as usize].is_empty()
            }
            _ => false // Conservative: treat other instructions as having dependencies
        }
    }

    fn update_dependencies(&mut self, inst: &Instruction) {
        // Update register dependencies based on instruction type
        match inst.opcode {
            Opcode::ADD | Opcode::SUB | Opcode::XOR | Opcode::OR | Opcode::AND |
            Opcode::SLL | Opcode::SRL | Opcode::SRA | Opcode::MUL => {
                let (rd, _, _) = inst.r_type();
                self.dependencies[rd as usize].clear();
            }
            _ => () // Other instructions don't update dependencies
        }
    }
}

const PREFETCH_BUFFER_SIZE: usize = 32;
const PREFETCH_STRIDE_TABLE_SIZE: usize = 16;

const SYSCALL_CACHE_SIZE: usize = 64;

#[derive(Debug, Clone, Default)]
struct PrefetchBuffer {
    entries: Box<[(u32, MemoryRecord); PREFETCH_BUFFER_SIZE]>,
    valid: Box<[bool; PREFETCH_BUFFER_SIZE]>,
    next_slot: usize,
}

impl PrefetchBuffer {
    fn lookup(&self, addr: u32) -> Option<&MemoryRecord> {
        for i in 0..PREFETCH_BUFFER_SIZE {
            if self.valid[i] && self.entries[i].0 == addr {
                return Some(&self.entries[i].1);
            }
        }
        None
    }

    fn insert(&mut self, addr: u32, record: MemoryRecord) {
        self.entries[self.next_slot] = (addr, record);
        self.valid[self.next_slot] = true;
        self.next_slot = (self.next_slot + 1) % PREFETCH_BUFFER_SIZE;
    }
}

#[derive(Debug, Clone, Default)]
struct StridePredictor {
    entries: Box<[(u32, i32); PREFETCH_STRIDE_TABLE_SIZE]>, // (last_addr, stride)
    next_slot: usize,
}

impl StridePredictor {
    fn predict_next_addr(&mut self, addr: u32) -> Option<u32> {
        // Look for matching entry
        for (last_addr, stride) in self.entries.iter() {
            if *last_addr != 0 {
                let predicted_stride = (addr as i32).wrapping_sub(*last_addr as i32);
                if predicted_stride == *stride {
                    // Update entry and return prediction
                    return Some(addr.wrapping_add(*stride as u32));
                }
            }
        }

        // No match found, create new entry
        self.entries[self.next_slot] = (addr, 0);
        self.next_slot = (self.next_slot + 1) % PREFETCH_STRIDE_TABLE_SIZE;
        None
    }
}

const BRANCH_PREDICTOR_SIZE: usize = 1024;
const BRANCH_PREDICTOR_MASK: u32 = (BRANCH_PREDICTOR_SIZE - 1) as u32;

const ICACHE_SIZE: usize = 1024;
const ICACHE_MASK: u32 = (ICACHE_SIZE - 1) as u32;

const NUM_REGISTERS: usize = 32;
const MEMORY_PAGE_SIZE: usize = 4096;
const MEMORY_PAGE_MASK: u32 = !(MEMORY_PAGE_SIZE as u32 - 1);

#[derive(Debug, Clone)]
struct SyscallCache {
    entries: Box<[(SyscallCode, Arc<dyn Syscall>); SYSCALL_CACHE_SIZE]>,
    valid: Box<[bool; SYSCALL_CACHE_SIZE]>,
}

impl Default for SyscallCache {
    fn default() -> Self {
        Self {
            entries: Box::new([(SyscallCode::HALT, Arc::new(crate::syscalls::halt::Halt)); SYSCALL_CACHE_SIZE]),
            valid: Box::new([false; SYSCALL_CACHE_SIZE]),
        }
    }
}

impl SyscallCache {
    fn lookup(&self, code: SyscallCode) -> Option<Arc<dyn Syscall>> {
        for i in 0..SYSCALL_CACHE_SIZE {
            if self.valid[i] && self.entries[i].0 == code {
                return Some(self.entries[i].1.clone());
            }
        }
        None
    }

    fn insert(&mut self, code: SyscallCode, syscall: Arc<dyn Syscall>) {
        // Simple FIFO replacement
        static mut NEXT_SLOT: usize = 0;
        unsafe {
            self.entries[NEXT_SLOT] = (code, syscall);
            self.valid[NEXT_SLOT] = true;
            NEXT_SLOT = (NEXT_SLOT + 1) % SYSCALL_CACHE_SIZE;
        }
    }
}

#[derive(Debug, Clone, Default)]
struct BranchPredictor {
    entries: Box<[(u32, bool); BRANCH_PREDICTOR_SIZE]>, // (pc, predicted_taken) pairs
}

impl BranchPredictor {
    fn predict(&self, pc: u32) -> bool {
        let idx = (pc & BRANCH_PREDICTOR_MASK) as usize;
        self.entries[idx].0 == pc && self.entries[idx].1
    }

    fn update(&mut self, pc: u32, taken: bool) {
        let idx = (pc & BRANCH_PREDICTOR_MASK) as usize;
        self.entries[idx] = (pc, taken);
    }
}

#[derive(Debug, Clone)]
struct InstructionCache {
    entries: Box<[(u32, Instruction); ICACHE_SIZE]>, // (pc, instruction) pairs
    valid: Box<[bool; ICACHE_SIZE]>,                 // Track which entries are valid
}

impl Default for InstructionCache {
    fn default() -> Self {
        Self {
            entries: Box::new([(0, Instruction::default()); ICACHE_SIZE]),
            valid: Box::new([false; ICACHE_SIZE]),
        }
    }
}

impl InstructionCache {
    fn lookup(&self, pc: u32) -> Option<Instruction> {
        let idx = (pc & ICACHE_MASK) as usize;
        if self.valid[idx] && self.entries[idx].0 == pc {
            Some(self.entries[idx].1)
        } else {
            None
        }
    }

    fn insert(&mut self, pc: u32, instruction: Instruction) {
        let idx = (pc & ICACHE_MASK) as usize;
        self.entries[idx] = (pc, instruction);
        self.valid[idx] = true;
    }
}

#[derive(Debug, Clone, Default)]
struct MemoryPage {
    values: Box<[MemoryRecord; MEMORY_PAGE_SIZE]>,
    initialized: HashSet<u32>, // Track which addresses are initialized
}

impl MemoryPage {
    fn new() -> Self {
        Self {
            values: Box::new([MemoryRecord::default(); MEMORY_PAGE_SIZE]),
            initialized: HashSet::new(),
        }
    }
}

/// Holds data describing the current state of a program's execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct ExecutionState {
    /// Instruction scheduler for improved ILP
    #[serde(skip)]
    pub scheduler: InstructionScheduler,

    /// Prefetch buffer for memory access optimization
    #[serde(skip)]
    pub prefetch_buffer: PrefetchBuffer,

    /// Stride predictor for memory access patterns
    #[serde(skip)]
    pub stride_predictor: StridePredictor,

    /// Syscall cache for faster syscall dispatch
    #[serde(skip)]
    pub syscall_cache: SyscallCache,

    /// Instruction cache for faster instruction fetch
    #[serde(skip)]
    pub icache: InstructionCache,

    /// Branch predictor for better branch handling
    #[serde(skip)]
    pub branch_predictor: BranchPredictor,

    /// The program counter.
    pub pc: u32,

    /// The shard clock keeps track of how many shards have been executed.
    pub current_shard: u32,

    /// Fixed array for registers (x0-x31) with hot/cold split
    pub hot_registers: [MemoryRecord; 8],  // Most frequently used registers (x0-x7)
    pub cold_registers: [MemoryRecord; 24], // Less frequently used registers (x8-x31)
    
    /// Register usage tracking for adaptive optimization
    #[serde(skip)]
    pub register_access_count: [u32; NUM_REGISTERS],

    /// Memory pages for fast access to frequently used memory regions
    #[serde(skip)]
    pub memory_pages: HashMap<u32, MemoryPage>,

    /// Fallback memory for sparse/infrequently accessed addresses
    pub memory: HashMap<u32, MemoryRecord>,

    /// Cache of recently accessed memory pages
    #[serde(skip)]
    pub page_cache: HashSet<u32>,

    /// The global clock keeps track of how many instructions have been executed through all shards.
    pub global_clk: u64,

    /// The clock increments by 4 (possibly more in syscalls) for each instruction that has been
    /// executed in this shard.
    pub clk: u32,

    /// Uninitialized memory addresses that have a specific value they should be initialized with.
    /// `SyscallHintRead` uses this to write hint data into uninitialized memory.
    pub uninitialized_memory: HashMap<u32, u32>,

    /// A stream of input values (global to the entire program).
    pub input_stream: Vec<Vec<u8>>,

    /// A ptr to the current position in the input stream incremented by `HINT_READ` opcode.
    pub input_stream_ptr: usize,

    /// A ptr to the current position in the proof stream, incremented after verifying a proof.
    pub proof_stream_ptr: usize,

    /// A stream of public values from the program (global to entire program).
    pub public_values_stream: Vec<u8>,

    /// A ptr to the current position in the public values stream, incremented when reading from
    /// `public_values_stream`.
    pub public_values_stream_ptr: usize,

    /// Keeps track of how many times a certain syscall has been called.
    pub syscall_counts: HashMap<SyscallCode, u64>,
}

impl ExecutionState {
    #[must_use]
    /// Create a new [`ExecutionState`].
    pub fn new(pc_start: u32) -> Self {
        Self {
            global_clk: 0,
            // Start at shard 1 since shard 0 is reserved for memory initialization.
            current_shard: 1,
            clk: 0,
            pc: pc_start,
            hot_registers: [MemoryRecord::default(); 8],
            cold_registers: [MemoryRecord::default(); 24],
            register_access_count: [0; NUM_REGISTERS],
            memory_pages: HashMap::new(),
            memory: HashMap::new(),
            page_cache: HashSet::new(),
            uninitialized_memory: HashMap::new(),
            input_stream: Vec::new(),
            input_stream_ptr: 0,
            public_values_stream: Vec::new(),
            public_values_stream_ptr: 0,
            proof_stream_ptr: 0,
            syscall_counts: HashMap::new(),
        }
    }

    /// Get a register value with access tracking
    pub fn get_register(&mut self, reg: usize) -> &MemoryRecord {
        self.register_access_count[reg] += 1;
        if reg < 8 {
            &self.hot_registers[reg]
        } else {
            &self.cold_registers[reg - 8]
        }
    }

    /// Set a register value
    pub fn set_register(&mut self, reg: usize, value: MemoryRecord) {
        if reg < 8 {
            self.hot_registers[reg] = value;
        } else {
            self.cold_registers[reg - 8] = value;
        }
    }

    /// Get a memory page, creating it if it doesn't exist
    fn get_or_create_page(&mut self, addr: u32) -> &mut MemoryPage {
        let page_addr = addr & MEMORY_PAGE_MASK;
        self.page_cache.insert(page_addr);
        self.memory_pages.entry(page_addr).or_insert_with(MemoryPage::new)
    }

    /// Get a memory record, checking the paged memory first
    pub fn get_memory(&mut self, addr: u32) -> Option<&MemoryRecord> {
        let page_addr = addr & MEMORY_PAGE_MASK;
        let offset = (addr & !MEMORY_PAGE_MASK) as usize;
        
        if let Some(page) = self.memory_pages.get(&page_addr) {
            if page.initialized.contains(&addr) {
                return Some(&page.values[offset]);
            }
        }
        
        self.memory.get(&addr)
    }

    /// Set a memory record, using paged memory for frequently accessed regions
    pub fn set_memory(&mut self, addr: u32, record: MemoryRecord) {
        let page_addr = addr & MEMORY_PAGE_MASK;
        let offset = (addr & !MEMORY_PAGE_MASK) as usize;

        if self.page_cache.contains(&page_addr) {
            let page = self.get_or_create_page(addr);
            page.values[offset] = record;
            page.initialized.insert(addr);
        } else {
            self.memory.insert(addr, record);
        }
    }
}

/// Holds data to track changes made to the runtime since a fork point.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ForkState {
    /// The `global_clk` value at the fork point.
    pub global_clk: u64,
    /// The original `clk` value at the fork point.
    pub clk: u32,
    /// The original `pc` value at the fork point.
    pub pc: u32,
    /// All memory changes since the fork point.
    pub memory_diff: HashMap<u32, Option<MemoryRecord>>,
    // /// The original memory access record at the fork point.
    // pub op_record: MemoryAccessRecord,
    // /// The original execution record at the fork point.
    // pub record: ExecutionRecord,
    /// Whether `emit_events` was enabled at the fork point.
    pub executor_mode: ExecutorMode,
}

impl ExecutionState {
    /// Save the execution state to a file.
    pub fn save(&self, file: &mut File) -> std::io::Result<()> {
        let mut writer = std::io::BufWriter::new(file);
        bincode::serialize_into(&mut writer, self).unwrap();
        writer.flush()?;
        writer.seek(std::io::SeekFrom::Start(0))?;
        Ok(())
    }
}
