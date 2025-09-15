{
  description = "A flake that use nix to manage uv venv using uv2nix.";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";

    uv2nix.url = "github:pyproject-nix/uv2nix";
    uv2nix.inputs.nixpkgs.follows = "nixpkgs";

    pyproject-build-systems.url = "github:pyproject-nix/build-system-pkgs";
    pyproject-build-systems.inputs = {
      nixpkgs.follows = "nixpkgs";
      uv2nix.follows = "uv2nix";
      pyproject-nix.follows = "pyproject-nix";
    };

    uv2nix_hammer_overrides.url = "github:TyberiusPrime/uv2nix_hammer_overrides";
    uv2nix_hammer_overrides.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, ... }@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system: let
      inherit (inputs.nixpkgs) lib;
      pkgs = import inputs.nixpkgs {
        inherit system;
        config.allowUnfree = true;  # for cuda packages
      };
      python = pkgs.python313;
      projectName = "beatdetect";  # matches [project.name] in pyproject.toml

      # Load Project Workspace (parses pyproject.toml, uv.lock)
      workspace = inputs.uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      # Generate Nix Overlay from uv.lock
      uvLockedOverlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
        # Optionally customise PEP 508 environment
        # environ = {
        #   platform_release = "5.10.65";
        # };
      };

      overrides = pkgs.lib.composeExtensions (inputs.uv2nix_hammer_overrides.overrides_strict pkgs) (
        final: prev: {
          # additional overlays
          # torch 2.5.1 in uv2nix_hammer_overrides -> torch 2.6.0
          torch = prev.torch.overrideAttrs (old: {
            buildInputs = (old.buildInputs or []) ++ [pkgs.cudaPackages.cusparselt];
          });
        }
      );

      # Construct package set
      pythonSet =
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage inputs.pyproject-nix.build.packages {
          inherit python;
        }).overrideScope (lib.composeManyExtensions [
            inputs.pyproject-build-systems.overlays.default
            uvLockedOverlay
            overrides
        ]);

      # Create an overlay enabling editable mode for all local dependencies.
      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
      };

      # Override previous set with our editable overlay.
      editablePythonSet = pythonSet.overrideScope (
        lib.composeManyExtensions [
          editableOverlay

          # Apply fixups for building an editable package of workspace packages
          (final: prev: builtins.listToAttrs [{
            name = projectName;
            value = prev.${projectName}.overrideAttrs (old: {
              src = lib.fileset.toSource {
                root = old.src;
                # Filter the sources going into editable build (files available to build system)
                # so the editable package doesn't have to be rebuilt on every change.
                # readme, pyproject.toml and __init__.py is minimal requirement for hatchling to build
                fileset = lib.fileset.unions [
                  (old.src + /pyproject.toml)
                  (old.src + /README.md)
                  (old.src + /${projectName}/__init__.py)
                ];
              };

              # hatchling dependency
              nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem { editables = []; };
            });
          }])
        ]
      );


      # Enable all optional dependencies for development.
      virtualenv = editablePythonSet.mkVirtualEnv
        "${projectName}-dev-env"
        workspace.deps.all  # uses deps from pyproject.toml [project.dependencies]
      ;
    in {
      packages.default = pythonSet.mkVirtualEnv "${projectName}-env" workspace.deps.default;
      devShells.default = pkgs.mkShell {
        packages = [virtualenv] ++ (with pkgs; [
          uv
          rclone  # for remote mounting dataset
          graphviz
        ]);

        env = {
          # Don't create venv using uv
          UV_NO_SYNC = "1";

          # Force uv to use Python interpreter from venv
          UV_PYTHON = "${virtualenv}/bin/python";

          # Prevent uv from downloading managed Python's
          UV_PYTHON_DOWNLOADS = "never";
        };

        shellHook = ''
          # Undo dependency propagation by nixpkgs.
          unset PYTHONPATH

          # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
          export REPO_ROOT=$(git rev-parse --show-toplevel)
        '';
      };
    });
}
