import streamlit as st

class ModuleManager:
    def __init__(self):
        # Initialize session state if not exists
        if 'modules' not in st.session_state:
            st.session_state.modules = {
                'rag': False,
                'search': False,
                'sql': False,
                'jira': False
            }
    
    def enable_module(self, module_name: str):
        if module_name in st.session_state.modules:
            st.session_state.modules[module_name] = True
            print(f"Enabled module: {module_name}")
    
    def disable_module(self, module_name: str):
        if module_name in st.session_state.modules:
            st.session_state.modules[module_name] = False
            print(f"Disabled module: {module_name}")
    
    def is_enabled(self, module_name: str) -> bool:
        enabled = st.session_state.modules.get(module_name, False)
        print(f"Module {module_name} enabled: {enabled}")
        return enabled
    
    @property
    def modules(self):
        return st.session_state.modules

# Global instance
module_manager = ModuleManager()
