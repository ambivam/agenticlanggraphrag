from dataclasses import dataclass
from typing import Dict, Optional
import os
from datetime import datetime

@dataclass
class ModuleConfig:
    enabled: bool = False
    output_file: str = ""

class ModuleManager:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.modules: Dict[str, ModuleConfig] = {
            "rag": ModuleConfig(),
            "mysql": ModuleConfig(),
            "web_search": ModuleConfig(),
            "jira": ModuleConfig()
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def enable_module(self, module_name: str):
        if module_name in self.modules:
            self.modules[module_name].enabled = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.modules[module_name].output_file = os.path.join(
                self.output_dir, 
                f"{module_name}_{timestamp}.txt"
            )
            
    def disable_module(self, module_name: str):
        if module_name in self.modules:
            self.modules[module_name].enabled = False
            
    def is_module_enabled(self, module_name: str) -> bool:
        return self.modules.get(module_name, ModuleConfig()).enabled
        
    def get_output_file(self, module_name: str) -> Optional[str]:
        if self.is_module_enabled(module_name):
            return self.modules[module_name].output_file
        return None
        
    def write_output(self, module_name: str, content: str):
        if not self.is_module_enabled(module_name):
            return
            
        output_file = self.get_output_file(module_name)
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().isoformat()}]\n")
                f.write(content)
                f.write("\n" + "-"*80 + "\n")
