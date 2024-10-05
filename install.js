module.exports = {
  run: [
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          xformers: true,
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "pip install -r requirements.txt",
        ]
      }
//    },
//    {
//      method: "fs.link",
//      params: {
//        venv: "env"
//      }
    }
  ]
}
