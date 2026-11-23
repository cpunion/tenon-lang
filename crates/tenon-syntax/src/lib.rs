pub mod ast;
pub mod lexer;
pub mod parser;
pub mod span;

pub use ast::*;
pub use parser::parse;
pub use span::{ParseError, Span};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_date_and_refinements() {
        let src = r#"
pub type Amount = Int64 where self > 0
pub type Odd = Int64 where self % 2 != 0
pub type Even = Int64 where self % 2 == 0

pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y Int, m Int where 1 <= self && self <= 12) Int
    ensures 28 <= result && result <= 31
{
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 }
        4 | 6 | 9 | 11 => 30
        _ => 31
    }
}

pub record Date {
    year  Int where 1 <= self && self <= 9999
    month Int where 1 <= self && self <= 12
    day   Int where 1 <= self && self <= days_in_month(year, month)
}
"#;
        let prog = parse(src).expect("should parse Date example");
        assert_eq!(prog.items.len(), 6);
    }

    #[test]
    fn parses_evolution_and_distributed_views() {
        let src = r#"
@evolve("billing.Order", storage = "orders")
pub record Order {
    @field(1) id Int64
    @field(2) amount Int64 where self > 0
}

evolution Order: v1 <-> v2 {
    archive ZeroOrder {
        order_id Int64
        amount Int64 where self == 0
    }
    up(old Order@v1) (Order@v2, Option[ZeroOrder]) {
        if old.amount == 0 {
            (Order@v2 { id = old.id, amount = 1 }, Some(ZeroOrder { order_id = old.id, amount = 0 }))
        } else {
            (Order@v2 { id = old.id, amount = old.amount }, None)
        }
    }
    down(new Order@v2, arch Option[ZeroOrder]) Order@v1
        retain(new.note)
    {
        Order@v1 { id = new.id, amount = new.amount }
    }
}

@sync_view("tasks.list")
view TasksListView from TaskAuthority {
    durability = Reconstructible
    bounds {
        max_items = 1000
        max_payload = 16 MiB
        max_frame = 4 MiB
        journal_budget = 16 MiB
        offline_recovery = BoundedReset
    }
}
"#;
        let prog = parse(src).expect("should parse evolution and sync view");
        assert_eq!(prog.items.len(), 3);
    }
}
