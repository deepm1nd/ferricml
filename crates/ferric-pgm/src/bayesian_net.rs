use petgraph::graph::DiGraph;

pub struct BayesianNetwork {
    pub graph: DiGraph<String, ()>,
}

impl BayesianNetwork {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
        }
    }
}
