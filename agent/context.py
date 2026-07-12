from agent.state import RepositoryState

def build_state_context(state: RepositoryState) -> str:

  sections = []

  if state.repo_path:
    sections.append(
      f"""
Repository path: {state.repo_path}
"""
    )

  if state.phase:
    sections.append(
      f"""
Current phase:
{state.phase}
"""
    )

  if state.important_files:
    sections.append(
      """
Important files discovered:
"""
      + "\n".join(f"- {file}" for file in state.important_files)    
    )

  if state.read_files:
    sections.append(
      """
Files already inspected:
"""
      + "\n".join(f"- {file}" for file in state.read_files)
    )

  if state.searched_keywords:
    sections.append(
      """
Keywords already searched:
"""
      + "\n".join(f"- {keyword}" for keyword in state.searched_keywords)
    )

  if state.errors:
    sections.append(
      """
Previous errors:
"""
      + "\n".join(f"- {error}" for error in state.errors)
    )

  if not sections:
    return """
No repository state available yet.
"""

  return ("""
Current repository analysis state:
"""  
          + 
          "\n\n".join(sections)
          +
          """
Use this information to avoid repeating unnecessary actions.
"""
          )