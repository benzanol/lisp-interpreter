pub mod lisp_objects {
    use std::sync::Mutex;
    use crate::lisp_global::LispContext;
    use crate::lisp_callables;

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
    }

    // type LispError = String;
    #[derive(Debug)]
    pub enum LispError {
        EvalEmptyList,
        UnboundSymbol(Lisp),
        NotCallable(Lisp),
        InvalidArguments(lisp_callables::ArgError),
        OutOfBounds(usize, Lisp),
    }

    pub type LispResult = Result<Lisp, LispError>;


    impl Lisp {
        pub fn print(&self, ctx: &LispContext) -> String {
            match self {
                Lisp::Num(n) => format!("{}", n),
                Lisp::Symbol(id) => ctx.symbols.get_symbol_name(*id),
                Lisp::Func(id) => format!("<callable:{}>", id),
                Lisp::String(text) => format!("\"{}\"", text),
                Lisp::List(id) =>
                    format!(
                        "({})",
                        ctx.lists
                            .get_vec(*id)
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
    use crate::lisp_objects::{Lisp, LispSymbol};
    use crate::lisp_callables::LispCallable;

    pub struct LispContext {
        pub symbols: SymbolTable,
        pub lists: ListTable,
        pub callables: Vec<LispCallable>,
    }

    impl LispContext {
        pub fn new() -> Self {
            let mut ctx = LispContext{
                symbols: SymbolTable::new(),
                lists: ListTable::new(),
                callables: crate::lisp_callables::get_callables()
            };

            // Put all of the callables into the symbol table
            for (idx, (name, _, _, _)) in ctx.callables.iter().enumerate() {
                let symbol = ctx.symbols.intern(name);
                ctx.symbols.set(symbol, Lisp::Func(idx));
            }

            ctx
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

            // Make nil and true self referencing symbols
            let nil_symbol = st.intern("nil");
            st.set(nil_symbol.clone(), nil_symbol);

            let true_symbol = st.intern("true");
            st.set(true_symbol.clone(), true_symbol);

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
        pub fn get_symbol_name(&self, id: usize) -> String {
            self.symbols.get(id).map(|s| s.name.clone()).unwrap()
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


        pub fn nil_sym  (&mut self) -> Lisp { self.intern("nil") }
        pub fn true_sym (&mut self) -> Lisp { self.intern("true") }
        pub fn quote_sym(&mut self) -> Lisp { self.intern("quote") }
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
        pub fn store(&mut self, vec: Vec<Lisp>) -> Lisp {
            Lisp::List(self.store_and_get_id(vec))
        }

        // Given a vec, add it to the global storage, and then return its id
        fn store_and_get_id(&mut self, list: Vec<Lisp>) -> usize {
            let id = self.next;
            self.lists.insert(id, list);
            self.next += 1;
            id
        }


        // Given a list id, return a clone of the vector
        pub fn get_vec(&self, id: usize) -> Vec<Lisp> {
            self.lists.get(&id).map(|r| r.clone()).unwrap()
        }

        pub fn get_length(&self, id: usize) -> usize {
            self.lists.get(&id).map(|l| l.len()).unwrap()
        }

        /// Given a list id and index, return the element at that index
        ///
        /// The option is for whether the index exits; If the list
        /// does not exist the function will panic
        pub fn get_nth(&self, list: usize, index: usize) -> Option<Lisp> {
            self.lists
                .get(&list)
                .unwrap()
                .get(index)
                .map(|e| e.clone())
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
        let LispContext{lists: list_table, symbols: symbol_table, callables: _} = ctx;
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
                let quote_sym = ctx.symbols.quote_sym();
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
    use crate::lisp_objects::{Lisp, LispResult, LispError};
    use crate::lisp_global::*;
    use crate::lisp_callables::*;

    pub fn evaluate_expression(ctx: &mut LispContext, expr: Lisp) -> LispResult {
        match expr {
            // If it is a symbol, get its value
            Lisp::Symbol(id) => ctx.symbols
                .get(expr)
                .ok_or(LispError::UnboundSymbol(Lisp::Symbol(id))),

            // If it is a list, pass the arguments to call_expression
            Lisp::List(id) => {
                // Access the physical values inside the list
                let elems = ctx.lists.get_vec(id);

                match elems.split_first() {
                    None => Err(LispError::EvalEmptyList),
                    Some((f, args)) => call_expression(ctx, f.clone(), args)
                }
            }

            // All other expressions are self evaluating
            expression => Ok(expression)
        }
    }

    pub fn call_expression(ctx: &mut LispContext, callable: Lisp, args: &[Lisp]) -> LispResult {
        let callable_value = evaluate_expression(ctx, callable) ?;
        match &callable_value {
            Lisp::Num(_) => Err(LispError::NotCallable(callable_value)),
            Lisp::String(_) => Err(LispError::NotCallable(callable_value)),
            Lisp::Symbol(_) => Err(LispError::NotCallable(callable_value)),

            Lisp::List(lambda_id) => {
                let lambda_vec = ctx.lists.get_vec(*lambda_id);

                // Lambda expression must have atleast 2 arguments
                if lambda_vec.len() < 2 {
                    return Err(LispError::NotCallable(Lisp::List(*lambda_id)));
                }

                // Get arguments based on whether the first symbol is
                // lambda, Lambda, or something else (error)
                let args = {
                    if lambda_vec[0] == ctx.symbols.intern("lambda") {
                        evaluate_arguments(ctx, args) ?
                    } else if lambda_vec[0] == ctx.symbols.intern("Lambda") {
                        Vec::from(args)
                    } else {
                        return Err(LispError::NotCallable(Lisp::List(*lambda_id)));
                    }};

                // Make sure the second item in the list is a list (of arg names)
                if let Lisp::List(args_id) = lambda_vec[1] {
                    let arg_vars = ctx.lists.get_vec(args_id);

                    // Make sure the number of arguments is correct
                    if args.len() < arg_vars.len() {
                        return Err(LispError::InvalidArguments(
                            ArgError::TooShort{found: args.len() as u8, min: arg_vars.len() as u8}
                        ));
                    } else if args.len() > arg_vars.len() {
                        return Err(LispError::InvalidArguments(
                            ArgError::TooLong{found: args.len() as u8, max: arg_vars.len() as u8}
                        ));
                    }

                    // Pair arguments with corresponding variables
                    let bindings: Vec<(usize, Lisp)> =
                        arg_vars.iter().zip(args)
                        .map(|(var, arg)| match var {
                            Lisp::Symbol(id) => Ok((*id, arg.clone())),
                            _                => Err(LispError::NotCallable(callable_value.clone()))
                        }).collect::<Result<Vec<(usize, Lisp)>, LispError>>() ?;

                    // The expression is the remainder of the lambda expression
                    let expr_vec = [&[ctx.symbols.intern("progn")], &lambda_vec[2..]].concat();
                    let expr = ctx.lists.store(expr_vec);

                    // Evaluate the expression with the argument bindings
                    evaluate_with(ctx, expr, bindings)
                        
                } else {
                    Err(LispError::NotCallable(callable_value))
                }
            }

            Lisp::Func(id) => {
                let arg_spec = ctx.callables[*id].2.clone();
                let eval_args = ctx.callables[*id].1;

                // Maybe evaluate the arguments, only if the second
                // field of the callable is true
                let evaled_args = if eval_args {
                    evaluate_arguments(ctx, args) ?
                } else { args.to_vec() };

                // Return error if the arguments are invalid
                if let Err(arg_error) = arg_spec.satisfies(ctx, &evaled_args) {
                    Err(LispError::InvalidArguments(arg_error))
                } else {
                    ctx.callables[*id].3(ctx, &evaled_args)
                }
            }
        }
    }

    pub fn evaluate_with(ctx: &mut LispContext, expr: Lisp, binds: Vec<(usize, Lisp)>) -> LispResult {
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

    pub fn sequence_expressions(ctx: &mut LispContext, exprs: &[Lisp]) -> Lisp {
        // Create a single progn expression containing exprs
        let expr_vec = [&[ctx.symbols.intern("progn")], exprs].concat();
        ctx.lists.store(expr_vec)
    }

    fn evaluate_arguments(ctx: &mut LispContext, args: &[Lisp]) -> Result<Vec<Lisp>, LispError> {
        // Evaluate each argument and add it to a vector
        let mut evaled_args: Vec<Lisp> = Vec::with_capacity(args.len());

        // Can't use map, because this wouldn't allow
        // returning an error as soon as it is reached
        for e in args {
            // If the argument evaluates to an error, the whole
            // expression evaluates to an error, so return on the spot
            evaled_args.push(evaluate_expression(ctx, e.clone()) ?);
        }
        Ok(evaled_args)
    }
}

pub mod lisp_callables {
    use crate::lisp_objects::*;
    use crate::lisp_global::*;
    use crate::lisp_evaluation::*;
    

    #[derive(Clone, Debug)]
    pub enum LispType {
        AnyP,
        NumP,
        StringP,
        SymbolP,
        ListP(Box<LispType>),
        TupleP(Vec<LispType>),
    }

    impl LispType {
        pub fn satisfies(&self, ctx: &mut LispContext, obj: &Lisp) -> Result<(), TypeError> {
            use LispType::*;
            let matches = match self {
                AnyP     => true,
                NumP     => matches!(obj, Lisp::Num(_)),
                StringP  => matches!(obj, Lisp::String(_)),
                SymbolP  => matches!(obj, Lisp::Symbol(_)),
                ListP(b) =>
                    if let Lisp::List(id) = obj {
                        ctx.lists.get_vec(*id).iter()
                            .all(|e| b.satisfies(ctx, e).is_ok())
                    } else { false }

                TupleP(type_vec) =>
                    if let Lisp::List(id) = obj {
                        let obj_vec = ctx.lists.get_vec(*id);
                        type_vec.len() == obj_vec.len() && {
                            type_vec.iter().zip(obj_vec)
                                .all(|(t, o)| t.satisfies(ctx, &o).is_ok())
                        }
                    } else { false }
            };

            if matches {
                Ok(())
            } else {
                Err(TypeError(self.clone(), obj.clone()))
            }
        }
    }

    #[derive(Debug)]
    pub struct TypeError(LispType, Lisp);


    #[derive(Clone)]
    pub struct ArgSpec{
        types: Vec<LispType>,
        extend: bool
    }

    #[derive(Debug)]
    pub enum ArgError {
        TooLong{found: u8, max: u8},
        TooShort{found: u8, min: u8},
        Type{idx: u8, error: TypeError},
    }

    impl ArgSpec {
        pub fn satisfies(&self, ctx: &mut LispContext, args: &[Lisp]) -> Result<(), ArgError> {
            let ArgSpec{types, extend} = self;

            if args.len() < types.len() {
                Err(ArgError::TooShort{found: args.len() as u8, min: types.len() as u8})

            } else if !extend && args.len() > types.len() {
                Err(ArgError::TooLong{found: args.len() as u8, max: types.len() as u8})

            } else {
                // Make sure the argument types match the required types

                // Get the last type to repeat, or if there are no
                // types at all, simply return ok
                let last_type = match types.iter().last() {
                    Some(last) => last,
                    None => { return Ok(()) }
                };

                // Get the first type error, if there is one
                let first_type_error = types.iter()
                    .chain(std::iter::repeat(last_type)) // Repeat the last type for all remaining args
                    .zip(args).enumerate()
                    .find_map(
                        |(idx, (typ, arg))|
                        match typ.satisfies(ctx, arg) {
                            Ok(_) => None,
                            Err(error) => Some((idx, error))
                        });

                // Return an arg error if there was a type error
                match first_type_error {
                    None => Ok(()),
                    Some((idx, type_error)) =>
                        Err(ArgError::Type{
                            idx: idx as u8,
                            error: type_error
                        })
                }
            }
        }
    }


    pub type LispFunc = dyn Fn(&mut LispContext, &[Lisp]) -> LispResult;

    pub type LispCallable = (&'static str, bool, ArgSpec, &'static LispFunc);


    pub fn get_callables() -> Vec<LispCallable> {
        vec![
            (&"eval", true, ArgSpec{types: vec![AnyP], extend: false},
             &|ctx: &mut LispContext, args: &[Lisp]|
             evaluate_expression(ctx, args[0].clone())
            ),
            (&"call", true, ArgSpec{types: vec![AnyP], extend: true},
             &|ctx: &mut LispContext, args: &[Lisp]|
             call_expression(ctx, args[0].clone(), &args[1..])
            ),
            (&"quote", false, ArgSpec{types: vec![AnyP], extend: false},
             &|_: &mut LispContext, args: &[Lisp]|
             Ok(args[0].clone())
            ),
            (&"progn", true, ArgSpec{types: vec![AnyP], extend: true},
             &|_: &mut LispContext, args: &[Lisp]|
             Ok(args.last().unwrap().clone())
            ),
            (&"lambda", false, ArgSpec{types: vec![], extend: true},
             &|ctx: &mut LispContext, args: &[Lisp]|
             Ok(ctx.lists.store([&[ctx.symbols.intern("lambda")], args].concat()))
            ),
            (&"set", true, ArgSpec{types: vec![SymbolP, AnyP], extend: false},
             &|ctx: &mut LispContext, args: &[Lisp]|
             if let &[Lisp::Symbol(sym), value] = &args {
                 ctx.symbols.set(Lisp::Symbol(*sym), value.clone());
                 Ok(Lisp::Symbol(*sym))
             } else { panic!("{}", ARG_CHECKER_MSG) }
            ),
            (&"setq", false, ArgSpec{types: vec![SymbolP, AnyP], extend: false},
             &|ctx: &mut LispContext, args: &[Lisp]| {
                 if let Lisp::Symbol(sym) = args[0] {
                     let value = evaluate_expression(ctx, args[1].clone()) ?;
                     ctx.symbols.set(Lisp::Symbol(sym), value.clone());
                     Ok(Lisp::Symbol(sym))
                 } else { panic!("{}", ARG_CHECKER_MSG) }
             }
            ),
            (&"let", false, ArgSpec{
                // Has the form (let ((SYMBOL ANY) ...) ANY...)
                types: vec![ListP(Box::new(TupleP(vec![SymbolP, AnyP]))), AnyP],
                extend: true
            },
             &|ctx: &mut LispContext, args: &[Lisp]| {
                 let bind_vec = if let Lisp::List(id) = args[0] {
                     ctx.lists.get_vec(id)
                 } else { panic!("{}", ARG_CHECKER_MSG) };

                 // Destructure the lisp list into a vector of bindings
                 let bindings: Vec<(usize, Lisp)> =
                     bind_vec.iter().map(
                         |binding|
                         if let Lisp::List(id) = binding {
                             if let [Lisp::Symbol(sym), value] = &ctx.lists.get_vec(*id)[..] {
                                 (*sym, value.clone())

                             } else { panic!("{}", ARG_CHECKER_MSG) }
                         } else { panic!("{}", ARG_CHECKER_MSG) }
                     ).collect();
                 
                 let expr = sequence_expressions(ctx, &args[1..]);
                 evaluate_with(ctx, expr, bindings)
             }
            ),
            (&"fn", false, ArgSpec{types: vec![ListP(Box::new(SymbolP)), AnyP], extend: true},
             &|ctx: &mut LispContext, args: &[Lisp]|
             if let Lisp::Symbol(name_id) = args[0] {
                 // Create the vector representing the lambda expression
                 // of the function, and store it as a lisp list
                 let lambda_vec = [&[ctx.symbols.intern("lambda")], &args[1..]].concat();
                 let lambda_expr = ctx.lists.store(lambda_vec);

                 // Assign the function name to the lambda expression
                 ctx.symbols.set(Lisp::Symbol(name_id), lambda_expr);

                 // Return the function name
                 Ok(Lisp::Symbol(name_id))
             } else { panic!("{}", ARG_CHECKER_MSG) }
            ),
            (&"print", true, ArgSpec{types: vec![AnyP], extend: false},
             &|ctx: &mut LispContext, args: &[Lisp]| {
                 println!("> {}", args[0].print(ctx));
                 Ok(ctx.symbols.nil_sym())
             }
            ),
            (&"list", true, ArgSpec{types: vec![], extend: true},
             &|ctx: &mut LispContext, args: &[Lisp]|
             Ok(ctx.lists.store(Vec::from(args)))
            ),
            (&"nth", true, ArgSpec{types: vec![ListP(Box::new(AnyP)), NumP], extend: false},
             &|ctx: &mut LispContext, args: &[Lisp]|
             if let Lisp::List(id) = args[0] {
                 if let Lisp::Num(float) = args[1] {
                     let idx = float as usize;

                     // Return out of bounds if the index isn't in the list
                     ctx.lists.get_nth(id, idx)
                         .ok_or(LispError::OutOfBounds(idx, args[0].clone()))

                 } else { panic!("{}", ARG_CHECKER_MSG) }
             } else { panic!("{}", ARG_CHECKER_MSG) }
            ),
            (&"len", true, ArgSpec{types: vec![ListP(Box::new(AnyP))], extend: false},
             &|ctx: &mut LispContext, args: &[Lisp]|
             if let Lisp::List(id) = args[0] {
                 Ok(Lisp::Num(ctx.lists.get_length(id) as f64))
             } else { panic!("{}", ARG_CHECKER_MSG) }
            ),
            (&"if", false, ArgSpec{types: vec![AnyP, AnyP, AnyP], extend: true},
             &|ctx: &mut LispContext, args: &[Lisp]|
             if evaluate_expression(ctx, args[0].clone()) ? == ctx.symbols.nil_sym() {
                 let sequence = sequence_expressions(ctx, &args[2..]);
                 evaluate_expression(ctx, sequence)
             } else {
                 evaluate_expression(ctx, args[1].clone())
             }
            ),
            (&"while", false, ArgSpec{types: vec![ListP(Box::new(AnyP)), NumP], extend: false},
             &|ctx: &mut LispContext, args: &[Lisp]| {
                 while evaluate_expression(ctx, args[0].clone()) ? != ctx.symbols.intern("nil") {
                     for a in &args[1..] {
                         evaluate_expression(ctx, a.clone()) ?;
                     }
                 }
                 Ok(ctx.symbols.intern("nil"))
             }
            ),
            (&"not", true, ArgSpec{types: vec![AnyP], extend: false},
             &|ctx: &mut LispContext, args: &[Lisp]|
             if args[0] == ctx.symbols.nil_sym() {
                 Ok(ctx.symbols.true_sym())
             } else {
                 Ok(ctx.symbols.nil_sym())
             }
            ),
            (&"+", true, ArgSpec{types: vec![NumP], extend: true},
             &|_: &mut LispContext, args: &[Lisp]|
             Ok(Lisp::Num(
                 args.iter().map(|e| match e {
                     Lisp::Num(n) => *n,
                     _ => panic!("{}", ARG_CHECKER_MSG)
                 }).sum::<f64>()
             ))
            ),
            (&"-", true, ArgSpec{types: vec![NumP], extend: true},
             &|_: &mut LispContext, args: &[Lisp]|
             if let &[Lisp::Num(n)] = args {
                 // If its a single number, negate it
                 Ok(Lisp::Num(-n))

             } else if let Lisp::Num(mut total) = args[0] {
                 // If there are multiple numbers, subtract the remaining
                 // from the first
                 for a in &args[1..] { match a {
                     Lisp::Num(n) => { total -= n; }
                     _ => { panic!("{}", ARG_CHECKER_MSG); }
                 }}
                 Ok(Lisp::Num(total))

             } else { panic!("{}", ARG_CHECKER_MSG); }
            ),
            (&"=", true, ArgSpec{types: vec![AnyP, AnyP], extend: true},
             &|ctx: &mut LispContext, args: &[Lisp]|
             if args[1..].iter().all(|e| *e == args[0]) {
                 Ok(ctx.symbols.true_sym())
             } else {
                 Ok(ctx.symbols.nil_sym())
             }
            ),
        ]
    }

    // The function name, implementation, and whether to evaluate the
    // arguments before passing them to the function
    use LispType::*;

    const ARG_CHECKER_MSG: &str = "Arg checker failed to prevent invalid state.";
}


fn main() {
    use std::io;
    use std::io::prelude::*;

    use crate::lisp_global::*;
    use crate::lisp_parse::*;
    use crate::lisp_evaluation::*;

    let mut ctx = LispContext::new();

    // let parse = read_expr("((() 1) (abc) 3)", &mut ctx);
    // println!("{:?}", parse);
    // println!("{}", ctx);

    for maybe_line in io::stdin().lock().lines() {
        let line = maybe_line.unwrap();
        if let Ok(e) = read_expr(&line, &mut ctx) {

            let result = evaluate_expression(&mut ctx, e.clone());

            match result {
                Ok(out) => {

                    // println!("Reading: `{}`", &line);

                    // println!("\n------BEFORE------\n{}------------------\n", ctx);
                    garbage_collect(&mut ctx);
                    println!("\n------AFTER-------\n{}------------------\n", ctx.lists);

                    println!("Output = {}\n", out.print(&mut ctx));

                }
                Err(error) => { println!("Error: {:?}", error); }
            }
        }
    }
}

