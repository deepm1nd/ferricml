use crate::block::BasicBlock;
use crate::function::Function;
use crate::module::Module;
use crate::types::Type;
use crate::value::ValueId;

pub struct IRBuilder {
    module: Module,
    next_value_id: usize,
}

impl IRBuilder {
    pub fn new(name: String) -> Self {
        Self {
            module: Module {
                name,
                functions: vec![],
            },
            next_value_id: 0,
        }
    }

    pub fn next_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    pub fn build(self) -> Module {
        self.module
    }

    pub fn add_function(
        &mut self,
        name: String,
        inputs: Vec<Type>,
        outputs: Vec<Type>,
    ) -> &mut Function {
        let func = Function {
            name,
            inputs,
            outputs,
            blocks: vec![BasicBlock { operations: vec![] }],
        };
        self.module.functions.push(func);
        self.module.functions.last_mut().unwrap()
    }
}
