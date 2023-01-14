{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
    buildInputs = with pkgs; [
        gcc11
        cmake
        python310
        python310Packages.numpy
        python310Packages.scipy
        python310Packages.matplotlib
        python310Packages.ipywidgets
        python310Packages.ipykernel
    ];
}
