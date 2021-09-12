



struct Window{A,B}
    __relative_coordinates::A
    __width::Int32
    __height::Int32
    __matrix_clicked::B
    __matrix_unclicked::B
    __reward::Int32
    __clicked::Bool
    __reward_given::Bool
    #on_click_listener?
    __resettable::Bool
end

Adapt.@adapt_structure Window