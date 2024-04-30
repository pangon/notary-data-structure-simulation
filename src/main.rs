use rand::Rng;

fn main() {
    let results = simulate();

    println!("r,k,n,node_count,size,leaf_count,internal_count,depth_min,depth_max,depth_mean,mean_size"); // Intestazione CSV
    for (r, k, n, node_count, size, leaf_count, internal_count, depth_min, depth_max, depth_mean, mean_size) in results {
        println!("{},{},{},{},{},{},{},{},{},{},{}", r, k, n, node_count, size, leaf_count, internal_count, depth_min, depth_max, depth_mean, mean_size);
    }
}

fn simulate() -> Vec<(u32, u16, u32, usize, u64, u64, u64, u16, u16, f32, f64)> {
    let mut rows = vec![];
    let mut rng = rand::thread_rng();

    for r in [1, 2, 3] {
        for k in [1, 2, 4, 8] {
            for x in 3..8 {
                let n = 10_u32.pow(x);

                println!("starting r={} k={} n={}", r, k, n); // Intestazione CSV

                let mut tree = Tree::new(r, k);

                for _ in 0..n {
                    let ledger: u64 = rng.gen();
                    tree.insert(ledger);
                }

                let node_count = tree.node_count();
                let (leaf_count, leaf_size) = tree.leaf_count_and_size();
                let (internal_count, internal_size) = tree.internal_count_and_size();
                let size = leaf_size + internal_size;

                let (depth_mean, depth_min, depth_max, mean_size) = tree.hash_path_depth_mean_min_max_size();

                rows.push((2_u32.pow(r as u32), k, n, node_count, size, leaf_count, internal_count, depth_min, depth_max, depth_mean, mean_size));
            }
        }
    }
    rows
}


#[derive(Debug)]
struct Node {
    depth: u16,
    is_leaf: bool,
    children: [usize; 8],
    parent: usize,
    size_from_root: u64,
}

impl Node {
    fn element_count(& self) -> usize {
        let mut len = 0;
        while (len < 8) && (self.children[len] != 0) {
            len += 1;
        }
        len
    }

    fn children_count(& self, r: usize) -> usize {
        let mut len = 0;
        for i in 0..r {
            if self.children[i] != 0 { len += 1; }
        }
        len
    }
}


struct NodeBuilder {
    nodes: Vec<Node>,
}

impl NodeBuilder {
    fn new() -> Self {
        NodeBuilder {
            nodes: vec![],
        }
    }

    fn build(&mut self, depth: u16, is_leaf: bool, parent: usize) -> usize {
        let node = Node {
            depth,
            is_leaf,
            children: [0; 8],
            parent,
            size_from_root: 0,
        };
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }
}


struct Tree {
    max_children_power: u16,
    max_leaf_elements: u16,
    node_builder: NodeBuilder,
    leaf_size_base: u64,
    internal_size_base: u64,
    hash_size: u64,
    r: usize,
}

impl Tree {
    fn new(max_children_power: u16, max_leaf_elements: u16) -> Self {
        let mut node_builder = NodeBuilder::new();
        node_builder.build(0, true, 0);
        Tree {
            max_children_power,
            max_leaf_elements,
            node_builder,
            leaf_size_base: (max_leaf_elements as f64).log2().ceil() as u64,
            internal_size_base: 2_u64.pow(max_children_power as u32),
            hash_size: 512,
            r: 2_u16.pow(max_children_power as u32) as usize
        }
    }

    fn node_count(& self) -> usize {
        self.node_builder.nodes.len()
    }

    fn leaf_count_and_size(& self) -> (u64, u64) {
        let mut count = 0;
        let mut size = 0;
        for leaf in self.node_builder.nodes.iter().filter(|n| n.is_leaf) {
            count += 1;
            size += leaf.element_count() as u64;
        }
        size *= self.hash_size;
        size += count * self.leaf_size_base;
        (count, size)
    }

    fn internal_count_and_size(& self) -> (u64, u64) {
        let mut count = 0;
        let mut size = 0;
        for internal in self.node_builder.nodes.iter().filter(|n| !n.is_leaf) {
            count += 1;
            size += internal.children_count(self.r) as u64;
        }
        size *= self.hash_size;
        size += count * self.internal_size_base;
        (count, size)
    }

    fn hash_path_depth_mean_min_max_size(&mut self) -> (f32, u16, u16, f64) {
        for i in 0..(self.node_builder.nodes.len()) {
            self.node_builder.nodes[i].size_from_root = self.node_builder.nodes[self.node_builder.nodes[i].parent].size_from_root;
            if self.node_builder.nodes[i].is_leaf {
                self.node_builder.nodes[i].size_from_root += self.leaf_size_base + self.hash_size * (self.node_builder.nodes[i].element_count() as u64);
            }
            else {
                self.node_builder.nodes[i].size_from_root += self.internal_size_base + self.hash_size * (self.node_builder.nodes[i].children_count(self.r) as u64);
            }
        }
        let mut count = 0;
        let mut max_depth = 0;
        let mut min_depth = 65535;
        let mut sum_depth: u32 = 0;
        let mut sum_size: f64 = 0.0;
        for leaf in self.node_builder.nodes.iter().filter(|n| n.is_leaf) {
            let elements = leaf.element_count();
            count += elements;
            if leaf.depth > max_depth { max_depth = leaf.depth; }
            if leaf.depth < min_depth { min_depth = leaf.depth; }
            sum_depth += ((leaf.depth + 1) * (elements as u16)) as u32;
            sum_size += ((leaf.size_from_root + self.leaf_size_base + self.hash_size * (elements as u64)) * (elements as u64)) as f64;
        }
        ((sum_depth as f32) / (count as f32), min_depth, max_depth, sum_size / (count as f64))
    }

    /*
    fn print(& self) {
        for node in &self.node_builder.nodes {
            dbg!(node);
        }
    }
    */

    fn insert(&mut self, ledger: u64) {
        let mut cursor_index = 0;
        //let mut cursor = &self.node_builder.nodes[cursor_index];
        while !self.node_builder.nodes[cursor_index].is_leaf {
            let children_index = ledger_to_index(ledger, self.node_builder.nodes[cursor_index].depth * self.max_children_power, self.max_children_power);
            if self.node_builder.nodes[cursor_index].children[children_index] == 0 {
                // create new leaf
                self.node_builder.nodes[cursor_index].children[children_index] = self.node_builder.build(self.node_builder.nodes[cursor_index].depth + 1, true, cursor_index);
            }
            cursor_index = self.node_builder.nodes[cursor_index].children[children_index];
        }
        let element_count = self.node_builder.nodes[cursor_index].element_count();
        if element_count < self.max_leaf_elements as usize {
            // insert into leaf
            self.node_builder.nodes[cursor_index].children[element_count] = ledger as usize;
        }
        else {
            let mut elements: [usize; 8] = [0; 8];
            elements.copy_from_slice(&self.node_builder.nodes[cursor_index].children[0..8]);
            // upgrade leaf to internal node
            self.node_builder.nodes[cursor_index].is_leaf = false;
            for i in 0..8 {
                self.node_builder.nodes[cursor_index].children[i] = 0;
            }
            // re-insert all existing elements
            for i in 0..self.max_leaf_elements {
                self.insert(elements[i as usize] as u64);
            }
            // re-insert original ledger
            self.insert(ledger);
        }
    }
}

/*
let ledger: u32 = rng.gen();
dbg!(ledger);
println!("ledger {} : {:b}", ledger, ledger);
for i in 0..10 {
    let index = ledger_to_index(ledger, i*3, 3);
    println!("index {} : {:b}", i, index);
}
*/
fn ledger_to_index(ledger: u64, offset: u16, bits: u16) -> usize {
    if (offset + bits) > 64 { panic!("ledger_to_index out of parameter validity space {}, {}, {}", ledger, offset, bits); }
    ((ledger >> offset) & (0b11111111 >> (8 - bits))) as usize
}
