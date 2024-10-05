module.exports = {
  daemon: true,
  run:[
    {
      id: "start",
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          ALLOW_FILE_UPLOAD : true
        },
        message: [
          "python app.py",
        ],
        on: [{
          "event": "/http:\/\/\\S+/",   
          "done": true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
      }
    },
    {
      method: "proxy.start",
      params: {
        uri: "{{local.url}}",
        name: "Local Sharing"
      }
    }
  ]
}
