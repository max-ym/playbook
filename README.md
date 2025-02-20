# Build WASM package

Please make sure to have latest Rust compiler as this project uses some of the latest features.
It should work on 1.84 and newer versions.

You also need to install cargo wasm-pack as per
[wasm-pack website](https://rustwasm.github.io/wasm-pack/installer/).

```bash
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

Then you can build the WASM package in fake-server mode by running:

```bash
cd wasm-bind && wasm-pack build --target web -- --features fake_server
```

This will write the WASM package and related JS/TS files to `wasm-bind/pkg` directory.
