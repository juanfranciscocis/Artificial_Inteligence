% Define valid moves
travel(e, w).
travel(w, e).

% Possible moves
move([X, X, _, Cabbage], wolf, [Y, Y, _, Cabbage]) :- travel(X, Y).
move([X, Wolf, _, X], cabbage, [Y, Wolf, _, Y]) :- travel(X, Y).
move([X, Wolf, _, Cabbage], nothing, [Y, Wolf, _, Cabbage]) :- travel(X, Y).

% Safe conditions
safe([X, _, X, _]).   % Wolf and cabbage are on the same bank as farmer

% Define the solution
solve([e, e, e, e], []).
solve(State, [FirstMove | OtherMoves]) :-
    move(State, FirstMove, NextState),
    safe(NextState),
    solve(NextState, OtherMoves).
