{
  description = "A flake that use nix to manage uv venv using uv2nix.";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

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
  };

  outputs = {
    nixpkgs,
    flake-utils,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      inherit (nixpkgs) lib;
      pkgs = import nixpkgs {
        system = "x86_64-linux";        
        config.allowUnfree = true;  # for cuda packages
        config.cudaSupport = true;
      };

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel"; # or sourcePreference = "sdist";
        # Optionally customise PEP 508 environment
        # environ = {
        #   platform_release = "5.10.65";
        # };
      };

      # Extend generated overlay with build fixups
      #
      # Uv2nix can only work with what it has, and uv.lock is missing essential metadata to perform some builds.
      # This is an additional overlay implementing build fixups.
      # See:
      # - https://pyproject-nix.github.io/uv2nix/FAQ.html
      cudaLibs = with pkgs; [
        cudaPackages.cudnn
        cudaPackages.nccl
        cudaPackages.cusparselt
        cudaPackages.cudatoolkit
        linuxPackages.nvidia_x11
      ];
      pyprojectOverrides = final: prev: {
        # Implement build fixups here.
        # torch + torchaudio
        torch = prev.torch.overrideAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ cudaLibs;
          nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem { editables = []; };
        });
        torchaudio = prev.torchaudio.overrideAttrs (old: {
          autoPatchelfIgnoreMissingDeps = true;  # not installing optional dependencies like ffmpeg [en|de]coder
          preFixup = lib.optionals (!pkgs.stdenv.isDarwin) ''
            addAutoPatchelfSearchPath "${final.torch}/${final.python.sitePackages}/torch/lib"
          '';
        });
        nvidia-cusolver-cu12 = prev.nvidia-cusolver-cu12.overrideAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ cudaLibs;
        });
        nvidia-cusparse-cu12 = prev.nvidia-cusparse-cu12.overrideAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ cudaLibs;
        });

        # librosa
        numba = prev.numba.overrideAttrs (old: {
          autoPatchelfIgnoreMissingDeps = true;
        });
        soundfile = prev.soundfile.overrideAttrs (old: {
          postInstall = ''
            pushd "$out/${final.python.sitePackages}"
            substituteInPlace soundfile.py \
              --replace-warn "_find_library('sndfile')" "'${pkgs.libsndfile.out}/lib/libsndfile${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}'"
            popd
          '';
        });
      };

      python = pkgs.python313;
      # Construct package set
      pythonSet =
        # Use base package set from pyproject.nix builders
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope (lib.composeManyExtensions [
            pyproject-build-systems.overlays.default
            overlay
            pyprojectOverrides
        ]);

      # Enable all optional dependencies for development.
      virtualenv = pythonSet.mkVirtualEnv "beatdetect-dev-env" workspace.deps.all;
    in {
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
        # } // lib.optionalAttrs pkgs.stdenv.isLinux {
        #   # Python libraries often load native shared objects using dlopen(3).
        #   # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
        #   LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (cudaLibs ++ pkgs.pythonManylinuxPackages.manylinux1);
        #   XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}";
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
