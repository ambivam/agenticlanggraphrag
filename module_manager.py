class ModuleManager:
    def __init__(self):
        self.modules = {
            'rag': False,
            'search': False,
            'sql': False,
            'jira': False
        }
    
    def enable_module(self, module_name: str):
        if module_name in self.modules:
            self.modules[module_name] = True
    
    def disable_module(self, module_name: str):
        if module_name in self.modules:
            self.modules[module_name] = False
    
    def is_enabled(self, module_name: str) -> bool:
        return self.modules.get(module_name, False)

# Global instance
module_manager = ModuleManager()
