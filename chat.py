import argparse
from typing import List

from minivllm.config.config import Config
from minivllm.config.sampling import SamplingParams
from minivllm.engine.engine import Engine


class ChatSession:
    def __init__(self, engine: Engine, system_prompt: str = None, sampling_params: SamplingParams = None):
        self.engine = engine
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.sampling_params = sampling_params or SamplingParams()
        self.history: List[dict] = []
        self.history.append({"role": "system", "content": self.system_prompt})
    
    def build_model_input_tokens(self, user_message: str) -> str:
        """Format conversation history into a prompt"""
        
        messages = self.history + [{"role": "user", "content": user_message}]
        return self.engine.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        
    def run_command(self, command: str, args: List[str]):
        """Run a special command"""
        if command == "/help":
            self.print_help()
            return
        
        if command == "/clear":
            self.clear_history()
            print("\n[Conversation history cleared]\n")
            return
        
        if command == "/history":
            self.print_history()
            return
        
        if command == "/system":
            if not args:
                print("\n[Usage: /system <system prompt text>]\n")
                return
            self.system_prompt = " ".join(args)
            self.history = []
            self.history.append({"role": "system", "content": self.system_prompt})
            print(f"\n[System prompt set to: {self.system_prompt}]\n")
            return
        
        if command == "/maxtokens":
            if len(args) != 1 or not args[0].isdigit():
                print("\n[Usage: /maxtokens <num>]\n")
                return
            self.sampling_params.max_tokens = int(args[0])
            print(f"\n[Max tokens set to {self.sampling_params.max_tokens}]\n")
            return
        
        print(f"\n[Unknown command: {command}. Type /help for available commands]\n")


    def print_help(self):
        """Print help message"""
        help_text = """
Commands:
/help     - Show this help message
/clear    - Clear conversation history
/exit     - Exit the chat
/reset    - Reset and clear history
/history  - Show conversation history

Settings:
/maxtokens <num>  - Set max tokens for response
/system <text>    - Set system prompt
"""
        print(help_text)

    def clear_history(self):
        """Clear conversation history"""
        self.history = []

    def print_history(self):
        """Print conversation history"""
        if not self.history:
            print("\n[No conversation history]\n")
            return
        
        print("\n" + "=" * 60)
        print("Conversation History:")
        print("=" * 60)
        for msg in self.history:
            role = msg["role"].capitalize()
            content = msg["content"]
            print(f"\n{role}: {content}")
        print("=" * 60 + "\n")


    def print_welcome(self):
        banner = """
╔═══════════════════════════════════════════════════════════╗
║                Mini-vLLM Interactive Chat                 ║
╚═══════════════════════════════════════════════════════════╝

Type your message and press Enter to chat.
Type /help for available commands.
Type /exit to exit.
    """
        print(banner)


    def print_settings(self):
        """Print current settings"""
        print(f"\nCurrent Settings:")
        print(f"Max Tokens: {self.sampling_params.max_tokens}")

        
    def start(self):
        self.print_welcome()
        self.print_settings()
        
    
        # Main chat loop
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    items = user_input.split()
                    cmd = items[0].lower()
                    args = items[1:]
                    
                    if cmd == "/exit":
                        print("\nGoodbye! 👋\n")
                        break
                    
                    self.run_command(cmd, args)
                    continue
                
                tokens = self.build_model_input_tokens(user_input)

                # Generate response
                print("Assistant: ", end="", flush=True)
                response = ""
                for text in self.engine.stream_generate(tokens, self.sampling_params):
                    print(text, end="", flush=True)
                    response += text
                print("\n")
                print("=" * 80 + "\n")

                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": response.strip()})
            
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋\n")
                break
            
            except EOFError:
                print("\n\nGoodbye! 👋\n")
                break
        


def main():
    parser = argparse.ArgumentParser(description="Mini-vLLM Interactive Chat CLI")
    
    # Model arguments
    parser.add_argument("--model", type=str, help="Local path of huggingface model", required=True)
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.", help="System prompt for the chat session")
    args = parser.parse_args()
    
    # Initialize engine
    print(f"Loading model: {args.model}...")
    config = Config(
        model=args.model,
    )
    engine = Engine(config)
    print("Model loaded successfully!\n")
    
    # Initialize chat session
    session = ChatSession(engine, system_prompt=args.system_prompt)
    
    session.start()
    
    
if __name__ == "__main__":
    main()
