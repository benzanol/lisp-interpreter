pub mod lisp_objects {
    use std::sync::Mutex;
    use crate::lisp_global::*;

    /// Note that since lists and symbols only store a reference, the
    /// id of the associated object in a list/symbol table, cloning the
    /// lisp object will clone the reference, and thus the clone will
    /// be pointing to the same exact list.
    #[derive(Clone, Debug, PartialEq)]
    pub enum Lisp {
        Num(f64),

        /// The index of a symbol in a SymbolTable
        Symbol(usize),

        /// The index of a list in a ListTable
        List(usize),

        /// Refers only to rust-level functions, as functions defined
        /// in lisp are stored as lists starting with the symbol lambda
        Func(usize),

        String(String),
        Error(String),
    }

    impl Lisp {

        pub fn print(&self, ctx: &LispContext) -> String {
            match self {
                Lisp::Num(n) => format!("{}", n),
                Lisp::Symbol(id) => ctx.symbols.get_symbol_name(*id).unwrap(),
                Lisp::Func(id) => format!("<callable:{}>", id),
                Lisp::String(text) => format!("\"{}\"", text),
                Lisp::Error(text) => format!("\"{}\"", text),
                Lisp::List(id) =>
                    format!(
                        "({})",
                        ctx.lists
                            .get_list_clone(*id)
                            .unwrap()
                            .iter()
                            .map(|e| e.print(ctx))
                            .collect::<Vec<String>>()
                            .join(" ")
                    )
            }
        }
    }

    #[derive(Debug)]
    pub struct LispSymbol {
        pub id: usize,
        pub name: String,
        pub object: Mutex<Option<Lisp>>
    }

}

pub mod lisp_global {
    use std::collections::HashMap;
    use std::collections::hash_map::Entry;
    use std::sync::Mutex;
    use std::fmt;
    use crate::lisp_objects::*;

    pub struct LispContext {
        pub symbols: SymbolTable,
        pub lists: ListTable
    }

    impl LispContext {
        pub fn new() -> Self {
            LispContext{
                symbols: SymbolTable::new(),
                lists: ListTable::new(),
            }
        }
    }

    impl fmt::Display for LispContext {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "Symbols:\n{}Lists:\n{}", self.symbols, self.lists)
        }
    }


    // Stores a vec of all interned symbols and their stored values
    #[derive(Debug)]
    pub struct SymbolTable {
        pub symbols: Vec<LispSymbol>,
        pub ids: HashMap<String, usize>
    }

    impl SymbolTable {
        pub fn new() -> SymbolTable {
            let mut st = SymbolTable{symbols: Vec::new(), ids: HashMap::new()};

            let nil_symbol = st.intern("nil");
            st.set(nil_symbol.clone(), nil_symbol);

            let t_symbol = st.intern("t");
            st.set(t_symbol.clone(), t_symbol);

            crate::lisp_callables::declare_callables(&mut st);
            st
        }

        // Given a symbol name, create it and return a reference to it
        pub fn intern(&mut self, name: &str) -> Lisp {
            Lisp::Symbol(self.get_symbol_id(name))
        }

        // Given a symbol name, return a lisp object of it, creating one if necessary
        fn get_symbol_id(&mut self, name: &str) -> usize {
            match self.ids.entry(name.to_string()) {
                Entry::Vacant(v) => {
                    // Set NAME's symbol number to ID
                    let id = self.symbols.len();
                    v.insert(id);

                    // Set the IDth symbol name to NAME
                    self.symbols.push(LispSymbol {
                        id: id,
                        name: name.to_string(),
                        object: Mutex::new(None)
                    });

                    id
                }
                Entry::Occupied(o) => *o.get()
            }
        }


        // Given a symbol id, return an optional string of its name
        pub fn get_symbol_name(&self, id: usize) -> Option<String> {
            self.symbols.get(id).map(|s| s.name.clone())
        }


        pub fn set(&mut self, symbol: Lisp, value: Lisp) {
            if let Lisp::Symbol(id) = symbol {
                let mutex = &mut self.symbols
                    .get_mut(id)
                    .expect("Symbol does not exist.")
                    .object;

                let mut guard = mutex.lock().unwrap();
                *guard = Some(value);

            } else {
                panic!("Can't set value of non-symbol")
            }
        }

        pub fn get(&self, symbol: Lisp) -> Option<Lisp> {
            if let Lisp::Symbol(id) = symbol {
                let mutex = &self.symbols
                    .get(id)
                    .expect("Symbol does not exist.")
                    .object;

                let guard = mutex.lock().unwrap();
                match guard.as_ref() {
                    Some(o) => Some(o.clone()),
                    None => None
                }

            } else {
                panic!("Can't get value of non-symbol")
            }
        }
    }

    impl fmt::Display for SymbolTable {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut out = String::new();
            for s in &self.symbols {
                out.push_str(&format!("{}: {} = {:?}\n", s.id, s.name, s.object));
            }
            write!(f, "{}", out)
        }
    }


    #[derive(Debug)]
    pub struct ListTable {
        lists: HashMap<usize, Vec<Lisp>>,
        next: usize // Id of last inserted list
    }

    impl ListTable {
        pub fn new() -> ListTable {
            ListTable{lists: HashMap::new(), next: 0}
        }

        // Given a vec, add it to the global storage, and then return a reference
        pub fn store(&mut self, list: Vec<Lisp>) -> Lisp {
            Lisp::List(self.get_list_id(list))
        }

        // Given a vec, add it to the global storage, and then return its id
        fn get_list_id(&mut self, list: Vec<Lisp>) -> usize {
            let id = self.next;
            self.lists.insert(id, list);
            self.next += 1;
            id
        }


        // Given a list id, return a clone of the vector
        pub fn get_list_clone(&self, id: usize) -> Option<Vec<Lisp>> {
            self.lists.get(&id).map(|r| r.clone())
        }

        pub fn get_list_length(&self, id: usize) -> Option<usize> {
            self.lists.get(&id).map(|l| l.len())
        }

        // Given a list id and index, return the element at that index
        pub fn get_list_element(&self, list: usize, element: usize) -> Option<Lisp> {
            self.lists.get(&list).map(|l| l.get(element).map(|e| e.clone())).flatten()
        }
    }

    impl fmt::Display for ListTable {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let mut out = String::new();
            for l in &self.lists {
                out.push_str(&format!("{}: {:?}\n", l.0, l.1));
            }
            write!(f, "{}", out)
        }
    }
    

    type RefTracker = HashMap<usize, bool>;

    pub fn garbage_collect(ctx: &mut LispContext) {
        let LispContext{lists: list_table, symbols: symbol_table} = ctx;
        let mut ref_tracker: RefTracker =
            list_table.lists.iter().map(|(n, _)| (*n, false)).collect();

        // Recursively mark every list referenced by a symbol, referenced
        // by a list referenced by a symbol, etc, as true in ref_tracker
        for symbol in &symbol_table.symbols {
            let guard = symbol.object.lock().unwrap();
            if let Some(Lisp::List(id)) = guard.as_ref() {
                track_references(&mut ref_tracker, &list_table, *id);
            }
        }

        // Remove all entries in the hash list_table which aren't referenced
        for cell in ref_tracker {
            // If the boolean is false, deallocate the list
            if !cell.1 {
                match list_table.lists.entry(cell.0) {
                    Entry::Occupied(o) => {
                        // println!("Deallocating list {} containing {:?}", cell.0, o.get());
                        o.remove_entry();
                    }
                    Entry::Vacant(_) => {
                        panic!("Tried to deallocate list which does not exist.");
                    }
                }
            }
        }

        println!("Storing {} lists.", list_table.lists.len());
    }

    fn track_references(ref_tracker: &mut RefTracker, list_table: &ListTable, id: usize) {
        if let Entry::Occupied(mut o) = ref_tracker.entry(id) {
            if !o.get() {
                // Mark the current list as referenced in the ref_tracker
                o.insert(true);

                // If id exists in ref_tracker, it will exist in list_table
                for obj in list_table.lists.get(&id).unwrap() {

                    // If the object is a list, mark it as referenced
                    if let Lisp::List(subid) = obj {
                        track_references(ref_tracker, &list_table, *subid);
                    }
                }
            }
        } else {
            // This means a symbol refers to a list that has already been garbage collected
            panic!("Index {} in reference tracker doesn't exist!", id);
        }
    }
}

pub mod lisp_parse{
    use std::collections::HashMap;
    use crate::lisp_objects::*;
    use crate::lisp_global::LispContext;


    pub enum ParseError {
        InvalidEscapedChar(char),
        EmptyExpression,
        UnclosedList,
        UnclosedString,
        MismatchedCloseParen,
    }
    
    type Parsed<'a> = Result<(Lisp, &'a str), ParseError>;


    pub fn read_expr(code: &str, ctx: &mut LispContext) -> Result<Lisp, ParseError> {
        parse_expr(code, ctx).map(|t| t.0)
    }


    fn parse_expr<'a>(code: &'a str, ctx: &mut LispContext) -> Parsed<'a> {
        // println!("Parsing: {}", code);

        let code = code.trim_start();

        match code.chars().next() {
            None => Err(ParseError::EmptyExpression),
            Some(')') => Err(ParseError::MismatchedCloseParen),
            Some('(') => {
                let (mut vec, rest) = parse_list_tail(&code[1..], ctx) ?;
                vec.reverse();
                Ok((ctx.lists.store(vec), rest))
            }
            Some('\'') => {
                let (expr, rest) = parse_expr(&code[1..], ctx) ?;
                let quote_sym = ctx.symbols.intern("quote");
                Ok((ctx.lists.store(vec![quote_sym, expr]), rest))
            }
            Some('"') => parse_string(code),
            Some(_) => parse_number_or_symbol(code, ctx)
        }
    }

    fn parse_list_tail<'a>(code: &'a str, ctx: &mut LispContext) -> Result<(Vec<Lisp>, &'a str), ParseError> {
        // Remove whitespace from the beginning of the list
        let code = code.trim_start();

        // If already finished
        match code.chars().next() {
            None => Err(ParseError::UnclosedList),
            Some(')') => Ok((Vec::new(), &code[1..])),

            _ => {
                // Read the next expression
                let (expr, rest) = parse_expr(code, ctx) ?;

                // Read the remaining text after the next expression
                let (mut vec, rest2) = parse_list_tail(rest, ctx) ?;

                vec.push(expr);
                Ok((vec, rest2))
            }
        }
    }

    fn parse_number_or_symbol<'a>(code: &'a str, ctx: &mut LispContext) -> Parsed<'a> {
        let code = code.trim_start();
        let non_symbol_chars = vec!['(', ')', '\'', '"', ' ', '\n', '\t'];
        
        // Advance idx until it is the position of the first non-symbol char
        let maybe_idx = code.chars()
            .position(|c| non_symbol_chars.contains(&c));

        // Split the string into the symbol text and remaining code
        let (text, rest) = match maybe_idx {
            Some(n) => code.split_at(n),
            None => (code, "")
        };

        if text == "" {
            // This should never happen, as this function should only
            // be called when there is a valid number/symbol character
            panic!("Tried to read empty symbol");

        } else if let Ok(n) = text.parse::<f64>() {
            // If it is a valid float
            Ok((Lisp::Num(n), rest))

        } else {
            // Intern and return the symbol
            let lisp_symbol = ctx.symbols.intern(text);
            Ok((lisp_symbol, rest))
        }
    }

    fn parse_string<'a>(code: &'a str) -> Parsed<'a> {
        let escaped_chars = [('\\', '\\'), ('"', '"'), ('n', '\n'), ('t', '\t')]
            .iter().map(|e| e.clone()).collect::<HashMap<char, char>>();

        let mut escape_next = false;
        let mut string = String::new();

        // Go through the characters in the code
        for (idx, c) in code[1..].chars().enumerate() {

            // If this character is escaped by a backslash
            if escape_next {
                if let Some(newchar) = escaped_chars.get(&c) {
                    string.push(*newchar);
                } else {
                    return Err(ParseError::InvalidEscapedChar(c));
                }

            } else {
                match c {
                    // The end of the string
                    '"' => { return Ok((Lisp::String(string), &code[1+idx..])); }
                    // Escape the next character
                    '\\' => { escape_next = true; }
                    // Add the current character to the vector
                    o => { string.push(o); }
                }
            }
        }
        // If never reached a closing ", return None
        Err(ParseError::UnclosedString)
    }
}

pub mod lisp_evaluation {
    use crate::lisp_objects::*;
    use crate::lisp_global::*;
    use crate::lisp_callables::*;

    pub fn evaluate_expression(ctx: &mut LispContext, expr: Lisp) -> Lisp {
        match expr {
            // If it is a symbol, get its value
            Lisp::Symbol(id) => ctx.symbols
                .get(expr)
                .or(Some(Lisp::Error(
                    format!("Symbol `{}` is not defined.",
                            ctx.symbols.get_symbol_name(id).unwrap()
                    )
                )))
                .unwrap(),

            // If it is a list, pass the arguments to call_expression
            Lisp::List(id) => {
                // Access the physical values inside the list
                let elems = ctx.lists
                    .get_list_clone(id)
                    .expect("Tried to access dropped list in `eval`.");

                match elems.split_first() {
                    None => Lisp::Error("Cannot evaluate empty list.".to_owned()),
                    Some((f, args)) => call_expression(ctx, f.clone(), args)
                }
            }

            // All other expressions are self evaluating
            expression => expression
        }
    }

    pub fn call_expression(ctx: &mut LispContext, callable: Lisp, args: &[Lisp]) -> Lisp {
        let callable_value = evaluate_expression(ctx, callable);
        match &callable_value {
            Lisp::Error(_) => callable_value,

            Lisp::Num(_) => Lisp::Error("Cannot call number.".to_owned()),
            Lisp::String(_) => Lisp::Error("Cannot call string.".to_owned()),
            Lisp::Symbol(_) => Lisp::Error("Cannot call symbol.".to_owned()),

            Lisp::List(lambda_id) => {
                let lambda_vec = ctx.lists.get_list_clone(*lambda_id).unwrap();

                if lambda_vec.len() < 2 {
                    return Lisp::Error(format!(
                        "Lambda expression must have atleast 2 elements, found {}.",
                        callable_value.print(ctx)
                    ));
                }

                let args =
                    if lambda_vec[0] == ctx.symbols.intern("lambda") {
                        match evaluate_arguments(ctx, args) {
                            Ok(arg_vec) => arg_vec,
                            Err(msg) => { return Lisp::Error(msg); }
                        }
                    } else if lambda_vec[0] == ctx.symbols.intern("Lambda") {
                        Vec::from(args)
                    } else {
                        return Lisp::Error(format!(
                            "Expected function, found {}.",
                            callable_value.print(ctx)
                        ));
                    };

                // Make sure the second item in the list is a list (of arg names)
                if let Lisp::List(args_id) = lambda_vec[1] {
                    let arg_vars = ctx.lists.get_list_clone(args_id).unwrap();

                    // Make sure the number of arguments is correct
                    if arg_vars.len() != args.len() {
                        return Lisp::Error(format!(
                            "Expected {} arguments, but found {}.",
                            arg_vars.len(), args.len()
                        ));
                    }

                    // Pair arguments with corresponding variables
                    let mut bindings: Vec<(usize, Lisp)> = Vec::with_capacity(args.len());
                    for i in 0..args.len() {
                        if let Lisp::Symbol(id) = arg_vars[i] {
                            bindings.push((id, args[i].clone()));
                        } else {
                            return Lisp::Error(format!(
                                "Lambda expression arguments must be symbols, found `{}`.",
                                arg_vars[i].print(ctx)
                            ));
                        }
                    }

                    // The expression is the remainder of the lambda expression
                    let expr_vec = [&[ctx.symbols.intern("progn")], &lambda_vec[2..]].concat();
                    let expr = ctx.lists.store(expr_vec);

                    // Evaluate the expression with the argument bindings
                    evaluate_with(ctx, expr, bindings)
                        
                } else {
                    return Lisp::Error("Second item of lambda expression must be a list.".to_owned());
                }
            }

            Lisp::Func(id) => {
                let tup = LISP_CALLABLES.get(*id)
                    .expect("Tried to access builtin callable which doesn't exist.");

                if !tup.3.satisfies(args.len() as u16) {
                    Lisp::Error(format!(
                        "Called `{}` with {} argument{}, but it requires {}.",
                        tup.0, args.len(), if args.len() == 1 {""} else {"s"}, tup.3
                    ))
                } else if tup.2 {
                    match evaluate_arguments(ctx, args) {
                        Ok(evaled_args) => tup.1(ctx, &evaled_args),
                        Err(msg) => Lisp::Error(msg)
                    }
                } else {
                    tup.1(ctx, args)
                }
            }
        }
    }

    pub fn evaluate_with(ctx: &mut LispContext, expr: Lisp, binds: Vec<(usize, Lisp)>) -> Lisp {
        // Set the new values in the context, and keep track of the
        // original values in a vector
        let mut originals: Vec<(usize, Option<Lisp>)> = Vec::with_capacity(binds.len());

        for (id, value) in binds {

            // Gain access to the symbol mutex directly to avoid the
            // symbol being changed between storing the original value
            // and setting the new one
            let symbol = &ctx.symbols.symbols[id];
            let mut guard = symbol.object.lock().unwrap();

            // Add the original value of the symbol to the `originals` vector
            originals.push((id, guard.clone()));

            // Set the new value
            *guard = Some(value);

        }


        // Run the code with the modified context
        let output = evaluate_expression(ctx, expr);


        // Reset the variables to their original values
        for (id, value) in originals {
            // Get access to the mutex
            let symbol = &ctx.symbols.symbols[id];
            let mut guard = symbol.object.lock().unwrap();

            // Set the original value
            *guard = value;
        }

        // Return the output of the expression
        output
    }

    fn evaluate_arguments(ctx: &mut LispContext, args: &[Lisp]) -> Result<Vec<Lisp>, String> {
        // Evaluate each argument and add it to a vector
        let mut evaled_args = Vec::with_capacity(args.len());

        // Can't use map, because this wouldn't allow
        // returning an error as soon as it is reached
        for e in args {
            let output = evaluate_expression(ctx, e.clone());

            // If the argument evaluates to an error, the whole
            // expression evaluates to an error, so return on the spot
            if let Lisp::Error(s) = output {
                return Err(s);
            } else {
                evaled_args.push(output);
            }
        }
        Ok(evaled_args)
    }
}

pub mod lisp_callables {
    use std::fmt;
    use crate::lisp_objects::*;
    use crate::lisp_global::*;
    use crate::lisp_evaluation::*;
    use ArgRange::*;
    

    pub enum ArgRange {
        Exactly(u16),
        Atleast(u16),
        Between(u16, u16),
    }

    impl ArgRange {
        pub fn satisfies(&self, count: u16) -> bool {
            match self {
                Exactly(n) => count == *n,
                Atleast(min) => count >= *min,
                Between(min, max) => count >= *min && count <= *max
            }
        }
    }

    impl fmt::Display for ArgRange {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                Exactly(n) => write!(f, "{}", n),
                Atleast(n) => write!(f, "atleast {}", n),
                Between(min, max) => write!(f, "{}-{}", min, max),
            }
        }
    }


    pub type LispCallable = dyn Fn(&mut LispContext, &[Lisp]) -> Lisp;

    /// The function name, implementation, and whether to evaluate the
    /// arguments before passing them to the function
    pub const LISP_CALLABLES: [(&str, &LispCallable, bool, ArgRange) ; 19] = [
        (&"eval", &lisp_eval, true, Exactly(1)),
        (&"call", &lisp_call, true, Atleast(1)),
        (&"quote", &lisp_quote, false, Exactly(1)),
        (&"progn", &lisp_progn, true, Atleast(1)),
        (&"lambda", &lisp_lambda, false, Atleast(0)),
        (&"set", &lisp_set, true, Exactly(2)),
        (&"setq", &lisp_setq, false, Exactly(2)),
        (&"let", &lisp_let, false, Atleast(2)),
        (&"fn", &lisp_fn, false, Atleast(3)),
        (&"print", &lisp_print, true, Exactly(1)),
        (&"list", &lisp_list, true, Atleast(0)),
        (&"nth", &lisp_nth, true, Exactly(2)),
        (&"len", &lisp_len, true, Exactly(1)),
        (&"if", &lisp_if, false, Atleast(3)),
        (&"while", &lisp_while, false, Atleast(1)),
        (&"not", &lisp_not, true, Exactly(1)),
        (&"=", &lisp_eq, true, Atleast(0)),
        (&"+", &lisp_plus, true, Atleast(0)),
        (&"-", &lisp_minus, true, Atleast(0)),
    ];

    pub fn declare_callables(st: &mut SymbolTable) {
        for (idx, (name, _, _, _)) in LISP_CALLABLES.iter().enumerate() {
            let symbol = st.intern(name);
            st.set(symbol, Lisp::Func(idx));
        }
    }


    fn lisp_eval(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        evaluate_expression(ctx, args[0].clone())
    }

    fn lisp_call(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        call_expression(ctx, args[0].clone(), &args[1..])
    }

    fn lisp_quote(_: &mut LispContext, args: &[Lisp]) -> Lisp {
        if args.len() == 1 {
            args[0].clone()
        } else {
            Lisp::Error("Quote requires 1 argument".to_owned())
        }
    }

    fn lisp_progn(_: &mut LispContext, args: &[Lisp]) -> Lisp {
        args.last().unwrap().clone()
    }

    fn lisp_lambda(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        ctx.lists.store([&[ctx.symbols.intern("lambda")], args].concat())
    }

    fn lisp_set(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        if let &[Lisp::Symbol(sym), value] = &args {
            ctx.symbols.set(Lisp::Symbol(*sym), value.clone());
            Lisp::Symbol(*sym)
        } else {
            Lisp::Error(format!("Can only set a symbol, not `{}`", args[0].print(ctx)))
        }
    }

    fn lisp_setq(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        let expr = evaluate_expression(ctx, args[1].clone());
        lisp_set(ctx, &[args[0].clone(), expr])
    }

    fn lisp_let(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        let bind_form = args[0].clone();
        let bind_vec = if let Lisp::List(id) = bind_form {
            ctx.lists.get_list_clone(id).unwrap()
        } else {
            return Lisp::Error(String::from(
                "First element of `let` expression must be a list of variable bindings."
            ));
        };

        if bind_vec.len() % 2 != 0 {
            return Lisp::Error(String::from(
                "Bind form must have an even number of elements."
            ));
        }

        let mut bindings: Vec<(usize, Lisp)> = Vec::with_capacity(bind_vec.len() / 2);
        for chunk in bind_vec.chunks(2) {
            if let Lisp::Symbol(id) = chunk[0] {
                bindings.push((id, chunk[1].clone()));
            } else {
                return Lisp::Error("Can't assign binding to non-symbol.".to_owned())
            }
        }

        let expr_vec = [&[ctx.symbols.intern("progn")], &args[1..]].concat();
        let expr = ctx.lists.store(expr_vec);

        evaluate_with(ctx, expr, bindings)
    }

    fn lisp_fn(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        if let Lisp::Symbol(name_id) = args[0] {
            if let Lisp::List(args_id) = args[1] {
                let arg_vec = ctx.lists.get_list_clone(args_id).unwrap();
                // If all elements in the list are symbols
                if arg_vec.iter().all(|a| if let Lisp::Symbol(_) = a {true} else {false}) {
                    let lambda_vec = [&[ctx.symbols.intern("lambda")], &args[1..]].concat();
                    let lambda_expr = ctx.lists.store(lambda_vec);
                    ctx.symbols.set(Lisp::Symbol(name_id), lambda_expr);
                    Lisp::Symbol(name_id)
                } else {
                    Lisp::Error("Second argument of `fn` must be a list of symbols.".to_owned())
                }
            } else {
                Lisp::Error("Second argument of `fn` must be a list of symbols.".to_owned())
            }
        } else {
            Lisp::Error("First argument of `fn` must be a symbol.".to_owned())
        }
    }

    fn lisp_print(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        println!("> {}", args[0].print(ctx));
        ctx.symbols.intern("nil")
    }

    fn lisp_list(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        ctx.lists.store(Vec::from(args))
    }

    fn lisp_nth(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        if let Lisp::List(id) = args[0] {
            if let Lisp::Num(float) = args[1] {
                let n = float as usize;
                let len = ctx.lists.get_list_length(id).unwrap();
                if n < len {
                    ctx.lists.get_list_element(id, n as usize).unwrap()
                } else {
                    Lisp::Error(format!("Index {} out of bounds for length {}", n, len))
                }
            } else {
                Lisp::Error(format!("Expected number, found {}", args[1].print(ctx)))
            }
        } else {
            Lisp::Error(format!("Expected list, found {}", args[0].print(ctx)))
        }
    }

    fn lisp_len(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        if let Lisp::List(id) = args[0] {
            Lisp::Num(ctx.lists.get_list_length(id).unwrap() as f64)
        } else {
            Lisp::Error(format!("Expected list, found {}", args[0].print(ctx)))
        }
    }
    
    fn lisp_if(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        if evaluate_expression(ctx, args[0].clone()) == ctx.symbols.intern("nil") {
            evaluate_expression(ctx, args[2].clone())
        } else {
            evaluate_expression(ctx, args[1].clone())
        }
    }

    fn lisp_while(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        while evaluate_expression(ctx, args[0].clone()) != ctx.symbols.intern("nil") {
            for a in &args[1..] {
                evaluate_expression(ctx, a.clone());
            }
        }
        ctx.symbols.intern("nil")
    }

    fn lisp_not(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        if args[0] == ctx.symbols.intern("nil") {
            ctx.symbols.intern("t")
        } else {
            ctx.symbols.intern("nil")
        }
    }

    fn lisp_plus(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        let mut sum = 0.0;
        for a in args {
            if let Lisp::Num(n) = a {
                sum += n;
            } else {
                return Lisp::Error(format!("Function `+` expected number, found `{}`", a.print(ctx)));
            }
        }
        Lisp::Num(sum)
    }

    fn lisp_minus(_: &mut LispContext, args: &[Lisp]) -> Lisp {
        if args.len() == 0 {
            Lisp::Num(0.0)

        } else if let &[Lisp::Num(n)] = args {
            Lisp::Num(-n)

        } else if let Lisp::Num(mut total) = args[0] {
            for a in &args[1..] {
                if let Lisp::Num(n) = a {
                    total -= n;
                } else {
                    return Lisp::Error("Function `-` must be called with numbers.".to_owned());
                }
            }
            Lisp::Num(total)

        } else {
            return Lisp::Error("Function `-` must be called with numbers.".to_owned())
        }
    }

    fn lisp_eq(ctx: &mut LispContext, args: &[Lisp]) -> Lisp {
        if args.len() == 0 {
            return ctx.symbols.intern("t");
        }

        for a in &args[1..] {
            if *a != args[0] {
                return ctx.symbols.intern("nil");
            }
        }
        ctx.symbols.intern("t")
    }
}


fn main() {
    use std::io;
    use std::io::prelude::*;

    use crate::lisp_global::*;
    use crate::lisp_parse::*;
    use crate::lisp_evaluation::*;

    let mut ctx = LispContext::new();
    let out_symbol = ctx.symbols.intern("$out");

    // let parse = read_expr("((() 1) (abc) 3)", &mut ctx);
    // println!("{:?}", parse);
    // println!("{}", ctx);

    for maybe_line in io::stdin().lock().lines() {
        let line = maybe_line.unwrap();
        if let Ok(e) = read_expr(&line, &mut ctx) {
            let out = evaluate_expression(&mut ctx, e.clone());
            ctx.symbols.set(out_symbol.clone(), out);
        }

        // println!("Reading: `{}`", &line);

        // println!("\n------BEFORE------\n{}------------------\n", ctx);
        garbage_collect(&mut ctx);
        println!("\n------AFTER-------\n{}------------------\n", ctx.lists);

        println!("Output = {}\n", ctx.symbols.get(out_symbol.clone()).unwrap().print(&mut ctx));

    }

    
}
