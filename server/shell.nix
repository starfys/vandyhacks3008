let
  pkgs-unstable = import <nixpkgs-unstable> {};
in
{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    buildInputs = [
      (pkgs-unstable.python3.withPackages (ps: with ps; [numpy opencv4 pillow sanic scipy]))
    ];
}
