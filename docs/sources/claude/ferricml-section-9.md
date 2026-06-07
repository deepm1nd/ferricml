        let mut idx = 0;
        let mut multiplier = 1;
        
        for i in (0..assignment.len()).rev() {
            idx += assignment[i] * multiplier;
            multiplier *= cardinalities[i];
        }
        
        idx
    }
}
```

---

## 4. Approximate Inference

### 4.1 Gibbs Sampling

```rust
pub struct GibbsSampling {
    /// Number of samples
    num_samples: usize,
    
    /// Burn-in period
    burn_in: usize,
    
    /// Thinning interval
    thinning: usize,
}

impl GibbsSampling {
    pub fn new(num_samples: usize, burn_in: usize, thinning: usize) -> Self {
        Self {
            num_samples,
            burn_in,
            thinning,
        }
    }
    
    pub fn infer(
        &self,
        network: &BayesianNetwork,
        query: &[NodeIndex],
        evidence: &Evidence,
    ) -> Result<Distribution> {
        use rand::thread_rng;
        let mut rng = thread_rng();
        
        // Initialize state
        let mut state = self.initialize_state(network, evidence, &mut rng)?;
        
        let mut samples = Vec::new();
        let total_iterations = self.burn_in + self.num_samples * self.thinning;
        
        for iteration in 0..total_iterations {
            // Update each non-evidence variable
            for &node in &network.topo_order {
                if evidence.contains_key(&node) {
                    continue;
                }
                
                // Sample from conditional distribution
                let conditional = self.compute_conditional(network, node, &state)?;
                let new_value = conditional.sample(&state, &mut rng);
                state.insert(node, new_value);
            }
            
            // Collect sample after burn-in
            if iteration >= self.burn_in && (iteration - self.burn_in) % self.thinning == 0 {
                let sample: Vec<Value> = query.iter()
                    .map(|&v| state[&v].clone())
                    .collect();
                samples.push(sample);
            }
        }
        
        // Compute empirical distribution
        self.samples_to_distribution(&samples, query, &network.variables)
    }
    
    fn initialize_state(
        &self,
        network: &BayesianNetwork,
        evidence: &Evidence,
        rng: &mut impl rand::Rng,
    ) -> Result<Evidence> {
        let mut state = evidence.clone();
        
        // Initialize non-evidence variables randomly
        for &node in &network.topo_order {
            if !state.contains_key(&node) {
                let var = &network.variables[&node];
                
                let value = match var.var_type {
                    VarType::Discrete => {
                        let card = var.cardinality.unwrap();
                        Value::Discrete(rng.gen_range(0..card))
                    }
                    VarType::Continuous => {
                        Value::Continuous(rng.gen_range(0.0..1.0))
                    }
                    VarType::Hybrid => Value::Missing,
                };
                
                state.insert(node, value);
            }
        }
        
        Ok(state)
    }
    
    fn compute_conditional(
        &self,
        network: &BayesianNetwork,
        node: NodeIndex,
        state: &Evidence,
    ) -> Result<Box<dyn CPD>> {
        // P(X | MarkovBlanket) ∝ P(X | Parents) * Π P(Child | Parents(Child))
        
        let cpd = &network.cpds[&node];
        
        // For simplicity, return the CPD itself
        // Full implementation would compute product of relevant factors
        Ok(cpd.clone())
    }
    
    fn samples_to_distribution(
        &self,
        samples: &[Vec<Value>],
        query: &[NodeIndex],
        variables: &HashMap<NodeIndex, Variable>,
    ) -> Result<Distribution> {
        // For discrete variables, compute empirical PMF
        if samples.is_empty() {
            return Err(Error::NoSamples);
        }
        
        // Assume single discrete query variable for simplicity
        let var = &variables[&query[0]];
        
        if let Some(card) = var.cardinality {
            let mut counts = vec![0; card];
            
            for sample in samples {
                if let Value::Discrete(v) = sample[0] {
                    counts[v] += 1;
                }
            }
            
            let total: usize = counts.iter().sum();
            let pmf: Vec<f64> = counts.iter()
                .map(|&c| c as f64 / total as f64)
                .collect();
            
            Ok(Distribution { pmf: Some(pmf), params: None })
        } else {
            // For continuous, compute mean and std
            let values: Vec<f64> = samples.iter()
                .filter_map(|s| {
                    if let Value::Continuous(v) = s[0] {
                        Some(v)
                    } else {
                        None
                    }
                })
                .collect();
            
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std = variance.sqrt();
            
            Ok(Distribution {
                pmf: None,
                params: Some(ContinuousParams::Gaussian { mean, std }),
            })
        }
    }
}
```

### 4.2 Variational Inference

```rust
pub struct MeanFieldVI {
    /// Maximum iterations
    max_iterations: usize,
    
    /// Convergence tolerance
    tolerance: f64,
}

impl MeanFieldVI {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
    
    pub fn infer(
        &self,
        network: &BayesianNetwork,
        query: &[NodeIndex],
        evidence: &Evidence,
    ) -> Result<Distribution> {
        // Initialize variational parameters
        let mut q_params = self.initialize_variational_params(network, evidence)?;
        
        for iteration in 0..self.max_iterations {
            let old_params = q_params.clone();
            
            // Update each variational factor
            for &node in &network.topo_order {
                if evidence.contains_key(&node) {
                    continue;
                }
                
                q_params.insert(node, self.update_variational_factor(
                    network,
                    node,
                    &q_params,
                )?);
            }
            
            // Check convergence
            if self.has_converged(&old_params, &q_params) {
                println!("Converged after {} iterations", iteration);
                break;
            }
        }
        
        // Extract distribution for query variables
        self.extract_query_distribution(query, &q_params)
    }
    
    fn initialize_variational_params(
        &self,
        network: &BayesianNetwork,
        evidence: &Evidence,
    ) -> Result<HashMap<NodeIndex, VariationalParams>> {
        let mut params = HashMap::new();
        
        for &node in &network.topo_order {
            if !evidence.contains_key(&node) {
                let var = &network.variables[&node];
                
                let vp = match var.var_type {
                    VarType::Discrete => {
                        let card = var.cardinality.unwrap();
                        VariationalParams::Categorical {
                            probs: vec![1.0 / card as f64; card],
                        }
                    }
                    VarType::Continuous => {
                        VariationalParams::Gaussian {
                            mean: 0.0,
                            std: 1.0,
                        }
                    }
                    VarType::Hybrid => return Err(Error::UnsupportedVarType),
                };
                
                params.insert(node, vp);
            }
        }
        
        Ok(params)
    }
    
    fn update_variational_factor(
        &self,
        network: &BayesianNetwork,
        node: NodeIndex,
        q_params: &HashMap<NodeIndex, VariationalParams>,
    ) -> Result<VariationalParams> {
        // Compute E[log p(x_i | x_{pa(i)})] under q
        // This is highly problem-specific; simplified implementation
        
        let var = &network.variables[&node];
        
        match var.var_type {
            VarType::Discrete => {
                let card = var.cardinality.unwrap();
                let mut log_probs = vec![0.0; card];
                
                // Sum contributions from CPD and children
                // Simplified: just use uniform
                for i in 0..card {
                    log_probs[i] = -(card as f64).ln();
                }
                
                // Convert log to probabilities and normalize
                let max_log = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probs: Vec<f64> = log_probs.iter()
                    .map(|&lp| (lp - max_log).exp())
                    .collect();
                
                let sum: f64 = probs.iter().sum();
                let normalized: Vec<f64> = probs.iter().map(|&p| p / sum).collect();
                
                Ok(VariationalParams::Categorical { probs: normalized })
            }
            VarType::Continuous => {
                // Update Gaussian parameters
                Ok(VariationalParams::Gaussian {
                    mean: 0.0,
                    std: 1.0,
                })
            }
            _ => Err(Error::UnsupportedVarType),
        }
    }
    
    fn has_converged(
        &self,
        old: &HashMap<NodeIndex, VariationalParams>,
        new: &HashMap<NodeIndex, VariationalParams>,
    ) -> bool {
        for (node, new_params) in new {
            if let Some(old_params) = old.get(node) {
                let diff = new_params.distance(old_params);
                if diff > self.tolerance {
                    return false;
                }
            }
        }
        true
    }
    
    fn extract_query_distribution(
        &self,
        query: &[NodeIndex],
        q_params: &HashMap<NodeIndex, VariationalParams>,
    ) -> Result<Distribution> {
        // Extract distribution for first query variable
        let node = query[0];
        let params = &q_params[&node];
        
        match params {
            VariationalParams::Categorical { probs } => {
                Ok(Distribution {
                    pmf: Some(probs.clone()),
                    params: None,
                })
            }
            VariationalParams::Gaussian { mean, std } => {
                Ok(Distribution {
                    pmf: None,
                    params: Some(ContinuousParams::Gaussian {
                        mean: *mean,
                        std: *std,
                    }),
                })
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum VariationalParams {
    Categorical { probs: Vec<f64> },
    Gaussian { mean: f64, std: f64 },
}

impl VariationalParams {
    fn distance(&self, other: &Self) -> f64 {
        match (self, other) {
            (
                VariationalParams::Categorical { probs: p1 },
                VariationalParams::Categorical { probs: p2 },
            ) => {
                p1.iter().zip(p2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum()
            }
            (
                VariationalParams::Gaussian { mean: m1, std: s1 },
                VariationalParams::Gaussian { mean: m2, std: s2 },
            ) => {
                (m1 - m2).abs() + (s1 - s2).abs()
            }
            _ => f64::INFINITY,
        }
    }
}
```

---

## 5. Markov Random Fields

### 5.1 Undirected Graphical Models

```rust
use petgraph::graph::UnGraph;

pub struct MarkovRandomField {
    /// Undirected graph
    graph: UnGraph<Variable, ()>,
    
    /// Potential functions (factors)
    potentials: Vec<Potential>,
    
    /// Variable metadata
    variables: HashMap<NodeIndex, Variable>,
}

pub struct Potential {
    /// Variables in scope
    scope: Vec<NodeIndex>,
    
    /// Potential values (unnormalized)
    values: Vec<f64>,
    
    /// Cardinalities
    cardinalities: Vec<usize>,
}

impl MarkovRandomField {
    pub fn new() -> Self {
        Self {
            graph: UnGraph::new_undirected(),
            potentials: Vec::new(),
            variables: HashMap::new(),
        }
    }
    
    pub fn add_variable(&mut self, var: Variable) -> NodeIndex {
        let node = self.graph.add_node(var.clone());
        self.variables.insert(node, var);
        node
    }
    
    pub fn add_edge(&mut self, v1: NodeIndex, v2: NodeIndex) {
        self.graph.add_edge(v1, v2, ());
    }
    
    pub fn add_potential(&mut self, potential: Potential) {
        self.potentials.push(potential);
    }
    
    pub fn probability(&self, assignment: &HashMap<NodeIndex, usize>) -> f64 {
        let unnormalized = self.unnormalized_probability(assignment);
        let z = self.partition_function();
        
        unnormalized / z
    }
    
    fn unnormalized_probability(&self, assignment: &HashMap<NodeIndex, usize>) -> f64 {
        self.potentials.iter()
            .map(|pot| pot.evaluate(assignment))
            .product()
    }
    
    fn partition_function(&self) -> f64 {
        // Sum over all assignments (intractable in general)
        // Use approximate methods
        unimplemented!("Partition function computation")
    }
    
    pub fn map_inference(&self) -> Result<HashMap<NodeIndex, usize>> {
        // Maximum a posteriori inference
        // Find assignment with highest probability
        unimplemented!("MAP inference")
    }
}

impl Potential {
    pub fn evaluate(&self, assignment: &HashMap<NodeIndex, usize>) -> f64 {
        // Convert assignment to index
        let values: Vec<usize> = self.scope.iter()
            .map(|&v| assignment.get(&v).copied().unwrap_or(0))
            .collect();
        
        let idx = Factor::assignment_to_index(&values, &self.cardinalities);
        self.values[idx]
    }
}
```

---

## 6. Structure Learning

### 6.1 Constraint-Based Learning

```rust
pub struct ConstraintBasedLearning {
    /// Significance level for independence tests
    alpha: f64,
}

impl ConstraintBasedLearning {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
    
    pub fn learn_structure(&self, data: &DataFrame) -> Result<BayesianNetwork> {
        let n_vars = data.n_columns();
        
        // Phase 1: Learn skeleton (undirected graph)
        let skeleton = self.learn_skeleton(data)?;
        
        // Phase 2: Orient edges
        let dag = self.orient_edges(skeleton, data)?;
        
        // Phase 3: Learn CPDs
        let mut network = BayesianNetwork::new();
        
        // Add variables
        let mut node_map = HashMap::new();
        for (i, var) in data.variables.iter().enumerate() {
            let node = network.add_variable(var.clone());
            node_map.insert(i, node);
        }
        
        // Add edges
        for edge in dag.edge_references() {
            network.add_edge(node_map[&edge.source()], node_map[&edge.target()])?;
        }
        
        // Learn CPDs from data
        self.learn_cpds(&mut network, data)?;
        
        Ok(network)
    }
    
    fn learn_skeleton(&self, data: &DataFrame) -> Result<UnGraph<usize, ()>> {
        let n_vars = data.n_columns();
        let mut graph = UnGraph::new_undirected();
        
        // Add nodes
        let nodes: Vec<_> = (0..n_vars).map(|i| graph.add_node(i)).collect();
        
        // Start with complete graph
        for i in 0..n_vars {
            for j in (i + 1)..n_vars {
                graph.add_edge(nodes[i], nodes[j], ());
            }
        }
        
        // Remove edges based on conditional independence tests
        let mut conditioning_set_size = 0;
        
        loop {
            let mut edges_to_remove = Vec::new();
            
            for edge in graph.edge_references() {
                let (i, j) = (edge.source().index(), edge.target().index());
                
                // Get neighbors
                let neighbors_i: HashSet<_> = graph.neighbors(nodes[i])
                    .filter(|&n| n != nodes[j])
                    .map(|n| n.index())
                    .collect();
                
                // Try all conditioning sets of current size
                for cond_set in Self::subsets(&neighbors_i, conditioning_set_size) {
                    if self.is_independent(data, i, j, &cond_set)? {
                        edges_to_remove.push((nodes[i], nodes[j]));
                        break;
                    }
                }
            }
            
            if edges_to_remove.is_empty() {
                conditioning_set_size += 1;
                
                // Stop if no node has enough neighbors
                let max_neighbors = graph.node_indices()
                    .map(|n| graph.neighbors(n).count())
                    .max()
                    .unwrap_or(0);
                
                if conditioning_set_size > max_neighbors {
                    break;
                }
            } else {
                for (i, j) in edges_to_remove {
                    graph.remove_edge(graph.find_edge(i, j).unwrap());
                }
            }
        }
        
        Ok(graph)
    }
    
    fn is_independent(
        &self,
        data: &DataFrame,
        i: usize,
        j: usize,
        conditioning: &[usize],
    ) -> Result<bool> {
        // Perform chi-square or G-test for independence
        // Simplified implementation
        let statistic = self.compute_chi_square(data, i, j, conditioning)?;
        let degrees_of_freedom = self.compute_dof(data, i, j, conditioning);
        
        use statrs::distribution::{ChiSquared, ContinuousCDF};
        let chi_sq = ChiSquared::new(degrees_of_freedom as f64).unwrap();
        let p_value = 1.0 - chi_sq.cdf(statistic);
        
        Ok(p_value > self.alpha)
    }
    
    fn compute_chi_square(
        &self,
        data: &DataFrame,
        i: usize,
        j: usize,
        conditioning: &[usize],
    ) -> Result<f64> {
        // Compute chi-square statistic
        unimplemented!("Chi-square test")
    }
    
    fn compute_dof(&self, data: &DataFrame, i: usize, j: usize, conditioning: &[usize]) -> usize {
        // Degrees of freedom
        let card_i = data.variables[i].cardinality.unwrap_or(2);
        let card_j = data.variables[j].cardinality.unwrap_or(2);
        let card_cond: usize = conditioning.iter()
            .map(|&k| data.variables[k].cardinality.unwrap_or(2))
            .product();
        
        (card_i - 1) * (card_j - 1) * card_cond
    }
    
    fn subsets(set: &HashSet<usize>, size: usize) -> Vec<Vec<usize>> {
        if size == 0 {
            return vec![vec![]];
        }
        
        if set.len() < size {
            return vec![];
        }
        
        let items: Vec<_> = set.iter().copied().collect();
        Self::combinations(&items, size)
    }
    
    fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
        if k == 0 {
            return vec![vec![]];
        }
        
        if items.is_empty() {
            return vec![];
        }
        
        let mut result = Vec::new();
        
        // Include first item
        for mut subset in Self::combinations(&items[1..], k - 1) {
            subset.insert(0, items[0]);
            result.push(subset);
        }
        
        // Exclude first item
        result.extend(Self::combinations(&items[1..], k));
        
        result
    }
    
    fn orient_edges(&self, skeleton: UnGraph<usize, ()>, data: &DataFrame) -> Result<DiGraph<usize, ()>> {
        // Convert to DAG using orientation rules
        unimplemented!("Edge orientation")
    }
    
    fn learn_cpds(&self, network: &mut BayesianNetwork, data: &DataFrame) -> Result<()> {
        for (&node, cpd) in network.cpds.iter_mut() {
            cpd.fit(data)?;
        }
        
        Ok(())
    }
}
```

---

## 7. Parameter Learning

### 7.1 Maximum Likelihood Estimation

```rust
pub struct MaximumLikelihoodEstimator;

impl MaximumLikelihoodEstimator {
    pub fn fit(network: &mut BayesianNetwork, data: &DataFrame) -> Result<()> {
        for (&node, cpd) in network.cpds.iter_mut() {
            cpd.fit(data)?;
        }
        
        Ok(())
    }
    
    pub fn log_likelihood(network: &BayesianNetwork, data: &DataFrame) -> Result<f64> {
        let mut log_lik = 0.0;
        
        for row in data.rows() {
            for (&node, cpd) in &network.cpds {
                let var = &network.variables[&node];
                let value = row.get(&var.name)?;
                
                let prob = cpd.probability(value, &row);
                log_lik += prob.ln();
            }
        }
        
        Ok(log_lik)
    }
}
```

---

## 8. Hidden Markov Models

### 8.1 HMM Implementation

```rust
pub struct HiddenMarkovModel {
    /// Number of hidden states
    n_states: usize,
    
    /// Number of observations
    n_observations: usize,
    
    /// Initial state probabilities
    initial: Vec<f64>,
    
    /// Transition matrix [from_state, to_state]
    transition: Vec<Vec<f64>>,
    
    /// Emission matrix [state, observation]
    emission: Vec<Vec<f64>>,
}

impl HiddenMarkovModel {
    pub fn new(n_states: usize, n_observations: usize) -> Self {
        let uniform_state = 1.0 / n_states as f64;
        let uniform_obs = 1.0 / n_observations as f64;
        
        Self {
            n_states,
            n_observations,
            initial: vec![uniform_state; n_states],
            transition: vec![vec![uniform_state; n_states]; n_states],
            emission: vec![vec![uniform_obs; n_observations]; n_states],
        }
    }
    
    pub fn forward(&self, observations: &[usize]) -> Result<Vec<Vec<f64>>> {
        let t_max = observations.len();
        let mut alpha = vec![vec![0.0; self.n_states]; t_max];
        
        // Initialize
        for s in 0..self.n_states {
            alpha[0][s] = self.initial[s] * self.emission[s][observations[0]];
        }
        
        // Recursion
        for t in 1..t_max {
            for s in 0..self.n_states {
                let mut sum = 0.0;
                for prev_s in 0..self.n_states {
                    sum += alpha[t - 1][prev_s] * self.transition[prev_s][s];
                }
                alpha[t][s] = sum * self.emission[s][observations[t]];
            }
        }
        
        Ok(alpha)
    }
    
    pub fn backward(&self, observations: &[usize]) -> Result<Vec<Vec<f64>>> {
        let t_max = observations.len();
        let mut beta = vec![vec![0.0; self.n_states]; t_max];
        
        // Initialize
        for s in 0..self.n_states {
            beta[t_max - 1][s] = 1.0;
        }
        
        // Recursion
        for t in (0..t_max - 1).rev() {
            for s in 0..self.n_states {
                let mut sum = 0.0;
                for next_s in 0..self.n_states {
                    sum += self.transition[s][next_s] 
                        * self.emission[next_s][observations[t + 1]]
                        * beta[t + 1][next_s];
                }
                beta[t][s] = sum;
            }
        }
        
        Ok(beta)
    }
    
    pub fn viterbi(&self, observations: &[usize]) -> Result<Vec<usize>> {
        let t_max = observations.len();
        let mut delta = vec![vec![0.0; self.n_states]; t_max];
        let mut psi = vec![vec![0; self.n_states]; t_max];
        
        // Initialize
        for s in 0..self.n_states {
            delta[0][s] = self.initial[s] * self.emission[s][observations[0]];
        }
        
        // Recursion
        for t in 1..t_max {
            for s in 0..self.n_states {
                let mut max_val = 0.0;
                let mut max_state = 0;
                
                for prev_s in 0..self.n_states {
                    let val = delta[t - 1][prev_s] * self.transition[prev_s][s];
                    if val > max_val {
                        max_val = val;
                        max_state = prev_s;
                    }
                }
                
                delta[t][s] = max_val * self.emission[s][observations[t]];
                psi[t][s] = max_state;
            }
        }
        
        // Backtrack
        let mut path = vec![0; t_max];
        path[t_max - 1] = delta[t_max - 1].iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        
        for t in (0..t_max - 1).rev() {
            path[t] = psi[t + 1][path[t + 1]];
        }
        
        Ok(path)
    }
}
```

---

## Summary

This section detailed probabilistic graphical models for FerricML:

**Key Components:**
1. **Bayesian Networks:** DAG structure with CPDs
2. **CPDs:** Tabular and Gaussian implementations
3. **Exact Inference:** Variable elimination algorithm
4. **Approximate Inference:** Gibbs sampling and variational inference
5. **MRFs:** Undirected graphical models
6. **Structure Learning:** Constraint-based methods
7. **Parameter Learning:** Maximum likelihood estimation
8. **HMMs:** Forward-backward and Viterbi algorithms

**Design Decisions:**
- Graph-based representation using petgraph
- Trait-based CPD system for extensibility
- Multiple inference algorithms for flexibility
- Support for both discrete and continuous variables

**Performance Considerations:**
- Variable elimination exponential in tree-width
- Sampling methods scale better but approximate
- Structure learning computationally expensive
- HMM algorithms O(T * N²) complexity

**Implementation Priority:**
1. Basic Bayesian network structure
2. Table CPD and exact inference
3. Sampling-based inference
4. HMM implementation
5. Structure learning (advanced)

This completes Section 9. One final section remains!# Section 9: Probabilistic Graphical Models

**FerricML Architecture Specification v3.0**  
**Word Count:** 3,200+ words  
**Implementation Priority:** Phase 3 - Advanced ML (Optional)

---

## Table of Contents

1. [Bayesian Network Foundation](#1-bayesian-network-foundation)
2. [Conditional Probability Distributions](#2-conditional-probability-distributions)
3. [Exact Inference](#3-exact-inference)
4. [Approximate Inference](#4-approximate-inference)
5. [Markov Random Fields](#5-markov-random-fields)
6. [Structure Learning](#6-structure-learning)
7. [Parameter Learning](#7-parameter-learning)
8. [Hidden Markov Models](#8-hidden-markov-models)

---

## 1. Bayesian Network Foundation

### 1.1 Network Structure

```rust
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Dfs;

pub struct BayesianNetwork {
    /// Directed acyclic graph structure
    graph: DiGraph<Variable, ()>,
    
    /// Conditional probability distributions
    cpds: HashMap<NodeIndex, Box<dyn CPD>>,
    
    /// Variable metadata
    variables: HashMap<NodeIndex, Variable>,
    
    /// Topological order for efficient inference
    topo_order: Vec<NodeIndex>,
}

#[derive(Debug, Clone)]
pub struct Variable {
    /// Variable name
    pub name: String,
    
    /// Variable type
    pub var_type: VarType,
    
    /// Cardinality (for discrete variables)
    pub cardinality: Option<usize>,
    
    /// Value range (for continuous variables)
    pub range: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarType {
    /// Discrete with finite states
    Discrete,
    
    /// Continuous real-valued
    Continuous,
    
    /// Hybrid (both discrete and continuous components)
    Hybrid,
}

impl BayesianNetwork {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            cpds: HashMap::new(),
            variables: HashMap::new(),
            topo_order: Vec::new(),
        }
    }
    
    pub fn add_variable(&mut self, var: Variable) -> NodeIndex {
        let node = self.graph.add_node(var.clone());
        self.variables.insert(node, var);
        node
    }
    
    pub fn add_edge(&mut self, parent: NodeIndex, child: NodeIndex) -> Result<()> {
        // Check for cycles
        if self.would_create_cycle(parent, child) {
            return Err(Error::CyclicGraph);
        }
        
        self.graph.add_edge(parent, child, ());
        
        // Recompute topological order
        self.compute_topological_order()?;
        
        Ok(())
    }
    
    pub fn set_cpd(&mut self, node: NodeIndex, cpd: Box<dyn CPD>) -> Result<()> {
        // Validate CPD matches variable type and parents
        let var = &self.variables[&node];
        let parents: Vec<_> = self.graph.neighbors_directed(node, petgraph::Direction::Incoming).collect();
        
        cpd.validate(var, &parents, &self.variables)?;
        
        self.cpds.insert(node, cpd);
        Ok(())
    }
    
    fn would_create_cycle(&self, from: NodeIndex, to: NodeIndex) -> bool {
        // Use DFS to check if path exists from 'to' to 'from'
        let mut dfs = Dfs::new(&self.graph, to);
        while let Some(node) = dfs.next(&self.graph) {
            if node == from {
                return true;
            }
        }
        false
    }
    
    fn compute_topological_order(&mut self) -> Result<()> {
        use petgraph::algo::toposort;
        
        self.topo_order = toposort(&self.graph, None)
            .map_err(|_| Error::CyclicGraph)?;
        
        Ok(())
    }
    
    pub fn check_independence(
        &self,
        x: NodeIndex,
        y: NodeIndex,
        z: &[NodeIndex],
    ) -> bool {
        // Check if X ⊥ Y | Z using d-separation
        self.d_separated(x, y, z)
    }
    
    fn d_separated(&self, x: NodeIndex, y: NodeIndex, z: &[NodeIndex]) -> bool {
        // Implement d-separation algorithm
        // Two nodes are d-separated given Z if all paths between them are blocked
        unimplemented!("d-separation algorithm")
    }
}
```

---

## 2. Conditional Probability Distributions

### 2.1 CPD Trait

```rust
pub trait CPD: Send + Sync {
    /// Compute P(X | Parents)
    fn probability(&self, value: &Value, evidence: &Evidence) -> f64;
    
    /// Sample from P(X | Parents)
    fn sample(&self, evidence: &Evidence, rng: &mut impl rand::Rng) -> Value;
    
    /// Get probability distribution over all values
    fn distribution(&self, evidence: &Evidence) -> Distribution;
    
    /// Learn parameters from data
    fn fit(&mut self, data: &DataFrame) -> Result<()>;
    
    /// Validate CPD matches variable and parents
    fn validate(&self, var: &Variable, parents: &[NodeIndex], vars: &HashMap<NodeIndex, Variable>) -> Result<()>;
}

pub type Evidence = HashMap<NodeIndex, Value>;

#[derive(Debug, Clone)]
pub enum Value {
    Discrete(usize),
    Continuous(f64),
    Missing,
}

pub struct Distribution {
    /// For discrete: probability mass function
    pub pmf: Option<Vec<f64>>,
    
    /// For continuous: parameters of distribution
    pub params: Option<ContinuousParams>,
}

#[derive(Debug, Clone)]
pub enum ContinuousParams {
    Gaussian { mean: f64, std: f64 },
    Exponential { lambda: f64 },
    Uniform { low: f64, high: f64 },
}
```

### 2.2 Tabular CPD (Discrete)

```rust
pub struct TableCPD {
    /// Variable node
    variable: NodeIndex,
    
    /// Parent nodes
    parents: Vec<NodeIndex>,
    
    /// Probability table
    /// Shape: [parent_1_card, parent_2_card, ..., variable_card]
    table: ndarray::ArrayD<f64>,
    
    /// Cardinalities
    variable_card: usize,
    parent_cards: Vec<usize>,
}

impl TableCPD {
    pub fn new(
        variable: NodeIndex,
        variable_card: usize,
        parents: Vec<NodeIndex>,
        parent_cards: Vec<usize>,
    ) -> Self {
        assert_eq!(parents.len(), parent_cards.len());
        
        // Initialize uniform distribution
        let mut shape = parent_cards.clone();
        shape.push(variable_card);
        
        let total_size: usize = shape.iter().product();
        let uniform_prob = 1.0 / variable_card as f64;
        
        let table = ndarray::ArrayD::from_elem(shape, uniform_prob);
        
        Self {
            variable,
            parents,
            table,
            variable_card,
            parent_cards,
        }
    }
    
    pub fn set_probabilities(&mut self, values: Vec<f64>) -> Result<()> {
        if values.len() != self.table.len() {
            return Err(Error::InvalidProbabilities);
        }
        
        self.table = ndarray::ArrayD::from_shape_vec(
            self.table.raw_dim(),
            values,
        )?;
        
        // Normalize each conditional distribution
        self.normalize()?;
        
        Ok(())
    }
    
    fn normalize(&mut self) -> Result<()> {
        // Normalize over last axis (variable values)
        let shape = self.table.shape();
        let n_conditions: usize = shape[..shape.len()-1].iter().product();
        
        for condition_idx in 0..n_conditions {
            let mut sum = 0.0;
            
            for value_idx in 0..self.variable_card {
                let mut index = self.flat_to_multi_index(condition_idx);
                index.push(value_idx);
                sum += self.table[&index[..]];
            }
            
            if sum > 0.0 {
                for value_idx in 0..self.variable_card {
                    let mut index = self.flat_to_multi_index(condition_idx);
                    index.push(value_idx);
                    self.table[&index[..]] /= sum;
                }
            }
        }
        
        Ok(())
    }
    
    fn flat_to_multi_index(&self, flat_idx: usize) -> Vec<usize> {
        let mut index = Vec::with_capacity(self.parents.len());
        let mut remaining = flat_idx;
        
        for &card in self.parent_cards.iter().rev() {
            index.push(remaining % card);
            remaining /= card;
        }
        
        index.reverse();
        index
    }
}

impl CPD for TableCPD {
    fn probability(&self, value: &Value, evidence: &Evidence) -> f64 {
        let Value::Discrete(var_value) = value else {
            return 0.0;
        };
        
        // Build index from parent evidence
        let mut index = Vec::with_capacity(self.parents.len() + 1);
        
        for &parent in &self.parents {
            match evidence.get(&parent) {
                Some(Value::Discrete(parent_value)) => index.push(*parent_value),
                _ => return 0.0,  // Missing parent evidence
            }
        }
        
        index.push(*var_value);
        
        self.table[&index[..]]
    }
    
    fn sample(&self, evidence: &Evidence, rng: &mut impl rand::Rng) -> Value {
        use rand::distributions::{Distribution as _, WeightedIndex};
        
        // Get distribution for given evidence
        let mut index = Vec::with_capacity(self.parents.len());
        
        for &parent in &self.parents {
            match evidence.get(&parent) {
                Some(Value::Discrete(parent_value)) => index.push(*parent_value),
                _ => return Value::Missing,
            }
        }
        
        // Get probabilities for this condition
        let mut probs = Vec::with_capacity(self.variable_card);
        for value_idx in 0..self.variable_card {
            let mut full_index = index.clone();
            full_index.push(value_idx);
            probs.push(self.table[&full_index[..]]);
        }
        
        // Sample
        let dist = WeightedIndex::new(&probs).unwrap();
        let sampled_value = dist.sample(rng);
        
        Value::Discrete(sampled_value)
    }
    
    fn distribution(&self, evidence: &Evidence) -> Distribution {
        let mut index = Vec::with_capacity(self.parents.len());
        
        for &parent in &self.parents {
            match evidence.get(&parent) {
                Some(Value::Discrete(parent_value)) => index.push(*parent_value),
                _ => return Distribution { pmf: None, params: None },
            }
        }
        
        let mut pmf = Vec::with_capacity(self.variable_card);
        for value_idx in 0..self.variable_card {
            let mut full_index = index.clone();
            full_index.push(value_idx);
            pmf.push(self.table[&full_index[..]]);
        }
        
        Distribution { pmf: Some(pmf), params: None }
    }
    
    fn fit(&mut self, data: &DataFrame) -> Result<()> {
        // Maximum likelihood estimation from data
        let mut counts = ndarray::ArrayD::zeros(self.table.raw_dim());
        
        for row in data.rows() {
            let mut index = Vec::with_capacity(self.parents.len() + 1);
            
            // Get parent values
            for &parent in &self.parents {
                let parent_var = &data.variables[parent];
                let value = row.get(&parent_var.name)?;
                
                if let Value::Discrete(v) = value {
                    index.push(*v);
                }
            }
            
            // Get variable value
            let var = &data.variables[self.variable];
            let value = row.get(&var.name)?;
            
            if let Value::Discrete(v) = value {
                index.push(*v);
                counts[&index[..]] += 1.0;
            }
        }
        
        // Convert counts to probabilities
        self.table = counts;
        self.normalize()?;
        
        Ok(())
    }
    
    fn validate(&self, var: &Variable, parents: &[NodeIndex], vars: &HashMap<NodeIndex, Variable>) -> Result<()> {
        if var.var_type != VarType::Discrete {
            return Err(Error::InvalidCPDType);
        }
        
        if parents.len() != self.parents.len() {
            return Err(Error::InvalidParents);
        }
        
        Ok(())
    }
}
```

### 2.3 Gaussian CPD (Continuous)

```rust
pub struct GaussianCPD {
    /// Variable node
    variable: NodeIndex,
    
    /// Parent nodes
    parents: Vec<NodeIndex>,
    
    /// Linear coefficients: mean = beta_0 + sum(beta_i * parent_i)
    beta: Vec<f64>,
    
    /// Variance
    variance: f64,
}

impl GaussianCPD {
    pub fn new(variable: NodeIndex, parents: Vec<NodeIndex>) -> Self {
        let n_params = parents.len() + 1;  // +1 for intercept
        
        Self {
            variable,
            parents,
            beta: vec![0.0; n_params],
            variance: 1.0,
        }
    }
}

impl CPD for GaussianCPD {
    fn probability(&self, value: &Value, evidence: &Evidence) -> f64 {
        let Value::Continuous(x) = value else {
            return 0.0;
        };
        
        // Compute mean from linear model
        let mut mean = self.beta[0];  // Intercept
        
        for (i, &parent) in self.parents.iter().enumerate() {
            match evidence.get(&parent) {
                Some(Value::Continuous(parent_value)) => {
                    mean += self.beta[i + 1] * parent_value;
                }
                _ => return 0.0,
            }
        }
        
        // Gaussian probability density
        let std = self.variance.sqrt();
        let coefficient = 1.0 / (std * (2.0 * std::f64::consts::PI).sqrt());
        let exponent = -((x - mean).powi(2)) / (2.0 * self.variance);
        
        coefficient * exponent.exp()
    }
    
    fn sample(&self, evidence: &Evidence, rng: &mut impl rand::Rng) -> Value {
        use rand_distr::{Distribution, Normal};
        
        // Compute mean
        let mut mean = self.beta[0];
        
        for (i, &parent) in self.parents.iter().enumerate() {
            match evidence.get(&parent) {
                Some(Value::Continuous(parent_value)) => {
                    mean += self.beta[i + 1] * parent_value;
                }
                _ => return Value::Missing,
            }
        }
        
        let std = self.variance.sqrt();
        let normal = Normal::new(mean, std).unwrap();
        
        Value::Continuous(normal.sample(rng))
    }
    
    fn distribution(&self, evidence: &Evidence) -> Distribution {
        let mut mean = self.beta[0];
        
        for (i, &parent) in self.parents.iter().enumerate() {
            match evidence.get(&parent) {
                Some(Value::Continuous(parent_value)) => {
                    mean += self.beta[i + 1] * parent_value;
                }
                _ => return Distribution { pmf: None, params: None },
            }
        }
        
        let std = self.variance.sqrt();
        
        Distribution {
            pmf: None,
            params: Some(ContinuousParams::Gaussian { mean, std }),
        }
    }
    
    fn fit(&mut self, data: &DataFrame) -> Result<()> {
        // Linear regression to estimate parameters
        let n_samples = data.n_rows();
        let n_features = self.parents.len();
        
        // Build design matrix X and target y
        let mut X = Vec::with_capacity(n_samples * (n_features + 1));
        let mut y = Vec::with_capacity(n_samples);
        
        for row in data.rows() {
            // Intercept
            X.push(1.0);
            
            // Parent values
            for &parent in &self.parents {
                let parent_var = &data.variables[parent];
                let value = row.get(&parent_var.name)?;
                
                if let Value::Continuous(v) = value {
                    X.push(*v);
                }
            }
            
            // Target value
            let var = &data.variables[self.variable];
            let value = row.get(&var.name)?;
            
            if let Value::Continuous(v) = value {
                y.push(*v);
            }
        }
        
        // Solve normal equations: beta = (X^T X)^{-1} X^T y
        let X_tensor = Tensor::from_vec(X, &[n_samples, n_features + 1]);
        let y_tensor = Tensor::from_vec(y, &[n_samples]);
        
        let XtX = X_tensor.transpose(0, 1)?.matmul(&X_tensor)?;
        let Xty = X_tensor.transpose(0, 1)?.matmul(&y_tensor.unsqueeze(1)?)?;
        
        let beta_tensor = XtX.inverse()?.matmul(&Xty)?;
        self.beta = beta_tensor.data().to_vec();
        
        // Estimate variance
        let predictions = X_tensor.matmul(&beta_tensor)?;
        let residuals = y_tensor.sub(&predictions.squeeze(1)?)?;
        let squared_residuals = residuals.pow(2.0)?;
        self.variance = squared_residuals.mean()? as f64;
        
        Ok(())
    }
    
    fn validate(&self, var: &Variable, parents: &[NodeIndex], vars: &HashMap<NodeIndex, Variable>) -> Result<()> {
        if var.var_type != VarType::Continuous {
            return Err(Error::InvalidCPDType);
        }
        
        Ok(())
    }
}
```

---

## 3. Exact Inference

### 3.1 Variable Elimination

```rust
pub struct VariableElimination {
    /// Elimination order
    elimination_order: EliminationOrder,
}

#[derive(Debug, Clone)]
pub enum EliminationOrder {
    /// Minimize fill-in edges
    MinFill,
    
    /// Minimize degree
    MinDegree,
    
    /// Custom order
    Custom(Vec<NodeIndex>),
}

impl VariableElimination {
    pub fn new(elimination_order: EliminationOrder) -> Self {
        Self { elimination_order }
    }
    
    pub fn infer(
        &self,
        network: &BayesianNetwork,
        query: &[NodeIndex],
        evidence: &Evidence,
    ) -> Result<Distribution> {
        // 1. Create factors from CPDs
        let mut factors = self.create_initial_factors(network, evidence)?;
        
        // 2. Determine elimination order
        let order = self.compute_elimination_order(network, query)?;
        
        // 3. Eliminate variables
        for &var in &order {
            if query.contains(&var) {
                continue;  // Don't eliminate query variables
            }
            
            factors = self.eliminate_variable(var, factors)?;
        }
        
        // 4. Multiply remaining factors and normalize
        let result = self.multiply_factors(&factors)?;
        let normalized = self.normalize_factor(&result)?;
        
        Ok(normalized)
    }
    
    fn create_initial_factors(
        &self,
        network: &BayesianNetwork,
        evidence: &Evidence,
    ) -> Result<Vec<Factor>> {
        let mut factors = Vec::new();
        
        for (&node, cpd) in &network.cpds {
            let factor = self.cpd_to_factor(node, cpd.as_ref(), network, evidence)?;
            factors.push(factor);
        }
        
        Ok(factors)
    }
    
    fn cpd_to_factor(
        &self,
        node: NodeIndex,
        cpd: &dyn CPD,
        network: &BayesianNetwork,
        evidence: &Evidence,
    ) -> Result<Factor> {
        let var = &network.variables[&node];
        let parents: Vec<_> = network.graph
            .neighbors_directed(node, petgraph::Direction::Incoming)
            .collect();
        
        // Create factor over variable and parents
        let mut scope = parents.clone();
        scope.push(node);
        
        // Remove evidence variables from scope
        scope.retain(|&v| !evidence.contains_key(&v));
        
        // Build factor values
        let factor = Factor::from_cpd(node, &scope, cpd, evidence, &network.variables)?;
        
        Ok(factor)
    }
    
    fn eliminate_variable(&self, var: NodeIndex, mut factors: Vec<Factor>) -> Result<Vec<Factor>> {
        // Find all factors mentioning var
        let (relevant, others): (Vec<_>, Vec<_>) = factors
            .into_iter()
            .partition(|f| f.scope.contains(&var));
        
        if relevant.is_empty() {
            return Ok(others);
        }
        
        // Multiply relevant factors
        let product = self.multiply_factors(&relevant)?;
        
        // Sum out var
        let marginalized = product.marginalize(var)?;
        
        // Add back to factor list
        let mut result = others;
        result.push(marginalized);
        
        Ok(result)
    }
    
    fn multiply_factors(&self, factors: &[Factor]) -> Result<Factor> {
        if factors.is_empty() {
            return Err(Error::EmptyFactorList);
        }
        
        let mut result = factors[0].clone();
        
        for factor in &factors[1..] {
            result = result.multiply(factor)?;
        }
        
        Ok(result)
    }
    
    fn normalize_factor(&self, factor: &Factor) -> Result<Distribution> {
        let sum: f64 = factor.values.iter().sum();
        
        if sum == 0.0 {
            return Err(Error::ZeroProbability);
        }
        
        let normalized: Vec<f64> = factor.values.iter().map(|&v| v / sum).collect();
        
        Ok(Distribution {
            pmf: Some(normalized),
            params: None,
        })
    }
    
    fn compute_elimination_order(
        &self,
        network: &BayesianNetwork,
        query: &[NodeIndex],
    ) -> Result<Vec<NodeIndex>> {
        match &self.elimination_order {
            EliminationOrder::Custom(order) => Ok(order.clone()),
            EliminationOrder::MinDegree => self.min_degree_order(network, query),
            EliminationOrder::MinFill => self.min_fill_order(network, query),
        }
    }
    
    fn min_degree_order(
        &self,
        network: &BayesianNetwork,
        query: &[NodeIndex],
    ) -> Result<Vec<NodeIndex>> {
        // Greedy: eliminate variable with smallest degree first
        let mut order = Vec::new();
        let mut remaining: HashSet<_> = network.graph.node_indices().collect();
        
        // Remove query variables
        for &q in query {
            remaining.remove(&q);
        }
        
        while !remaining.is_empty() {
            // Find variable with minimum degree
            let min_var = remaining.iter()
                .min_by_key(|&&v| {
                    network.graph.neighbors_undirected(v).count()
                })
                .copied()
                .unwrap();
            
            order.push(min_var);
            remaining.remove(&min_var);
        }
        
        Ok(order)
    }
    
    fn min_fill_order(
        &self,
        network: &BayesianNetwork,
        query: &[NodeIndex],
    ) -> Result<Vec<NodeIndex>> {
        // Greedy: eliminate variable that adds fewest edges
        unimplemented!("Min-fill heuristic")
    }
}

#[derive(Debug, Clone)]
pub struct Factor {
    /// Variables in factor scope
    scope: Vec<NodeIndex>,
    
    /// Cardinalities of variables
    cardinalities: Vec<usize>,
    
    /// Factor values (flattened multi-dimensional array)
    values: Vec<f64>,
}

impl Factor {
    pub fn from_cpd(
        node: NodeIndex,
        scope: &[NodeIndex],
        cpd: &dyn CPD,
        evidence: &Evidence,
        variables: &HashMap<NodeIndex, Variable>,
    ) -> Result<Self> {
        let cardinalities: Vec<usize> = scope.iter()
            .map(|&v| variables[&v].cardinality.unwrap_or(1))
            .collect();
        
        let total_size: usize = cardinalities.iter().product();
        let mut values = Vec::with_capacity(total_size);
        
        // Enumerate all assignments
        for assignment_idx in 0..total_size {
            let assignment = Self::index_to_assignment(assignment_idx, &cardinalities);
            
            // Build evidence for this assignment
            let mut full_evidence = evidence.clone();
            for (i, &var) in scope.iter().enumerate() {
                full_evidence.insert(var, Value::Discrete(assignment[i]));
            }
            
            let prob = cpd.probability(&full_evidence[&node], &full_evidence);
            values.push(prob);
        }
        
        Ok(Self {
            scope: scope.to_vec(),
            cardinalities,
            values,
        })
    }
    
    pub fn multiply(&self, other: &Factor) -> Result<Self> {
        // Compute union of scopes
        let mut new_scope = self.scope.clone();
        for &var in &other.scope {
            if !new_scope.contains(&var) {
                new_scope.push(var);
            }
        }
        
        // Build new cardinalities
        let mut new_cards = Vec::new();
        for &var in &new_scope {
            let card = if let Some(pos) = self.scope.iter().position(|&v| v == var) {
                self.cardinalities[pos]
            } else {
                other.cardinalities[other.scope.iter().position(|&v| v == var).unwrap()]
            };
            new_cards.push(card);
        }
        
        let total_size: usize = new_cards.iter().product();
        let mut new_values = Vec::with_capacity(total_size);
        
        // Compute product
        for idx in 0..total_size {
            let assignment = Self::index_to_assignment(idx, &new_cards);
            
            let val1 = self.get_value(&assignment, &new_scope);
            let val2 = other.get_value(&assignment, &new_scope);
            
            new_values.push(val1 * val2);
        }
        
        Ok(Self {
            scope: new_scope,
            cardinalities: new_cards,
            values: new_values,
        })
    }
    
    pub fn marginalize(&self, var: NodeIndex) -> Result<Self> {
        let var_pos = self.scope.iter().position(|&v| v == var)
            .ok_or(Error::VariableNotInScope)?;
        
        let var_card = self.cardinalities[var_pos];
        
        // New scope without var
        let mut new_scope = self.scope.clone();
        new_scope.remove(var_pos);
        
        let mut new_cards = self.cardinalities.clone();
        new_cards.remove(var_pos);
        
        if new_scope.is_empty() {
            // Marginalize to constant
            let sum: f64 = self.values.iter().sum();
            return Ok(Self {
                scope: vec![],
                cardinalities: vec![],
                values: vec![sum],
            });
        }
        
        let new_size: usize = new_cards.iter().product();
        let mut new_values = vec![0.0; new_size];
        
        // Sum over var
        for idx in 0..self.values.len() {
            let assignment = Self::index_to_assignment(idx, &self.cardinalities);
            let mut new_assignment = assignment.clone();
            new_assignment.remove(var_pos);
            
            let new_idx = Self::assignment_to_index(&new_assignment, &new_cards);
            new_values[new_idx] += self.values[idx];
        }
        
        Ok(Self {
            scope: new_scope,
            cardinalities: new_cards,
            values: new_values,
        })
    }
    
    fn get_value(&self, assignment: &[usize], full_scope: &[NodeIndex]) -> f64 {
        // Project assignment to this factor's scope
        let mut local_assignment = Vec::new();
        
        for &var in &self.scope {
            let pos = full_scope.iter().position(|&v| v == var).unwrap();
            local_assignment.push(assignment[pos]);
        }
        
        let idx = Self::assignment_to_index(&local_assignment, &self.cardinalities);
        self.values[idx]
    }
    
    fn index_to_assignment(mut idx: usize, cardinalities: &[usize]) -> Vec<usize> {
        let mut assignment = Vec::with_capacity(cardinalities.len());
        
        for &card in cardinalities.iter().rev() {
            assignment.push(idx % card);
            idx /= card;
        }
        
        assignment.reverse();
        assignment
    }
    
    fn assignment_to_index(assignment: &[usize], cardinalities: &[usize]) -> usize {
        let mut idx = 0;
        let mut multiplier = 1;
        