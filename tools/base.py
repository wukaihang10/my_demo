class Tool:
  def __init__(
      self,
      name,
      description,
      function,
      parameters
  ):
    self.name = name
    self.description = description
    self.function = function
    self.parameters = parameters

  def execute(self, **kwargs):
    return self.function(**kwargs)
  
  def schema(self):
    return {
      "type": "function",
      "function":
      {
        "name": self.name,
        "description": self.description,
        "parameters": self.parameters
      }
    }