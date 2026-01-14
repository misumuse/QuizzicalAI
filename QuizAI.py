"""
Advanced Self-Learning LLM with GUI
Intelligent responses, word embeddings, and knowledge integration

In Memory of Aaron Swartz, who wanted fair, free information for the world

Requirements:
pip install numpy requests tkinter
"""

import re
import pickle
import os
import threading
import time
import random
import numpy as np
import requests
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
from tkinter import ttk

class WordEmbedding:
    """Word embedding system for semantic understanding"""
    
    def __init__(self, dim=100):
        self.dim = dim
        self.word_to_vec = {}
        self.word_freq = {}
        
    def add_word(self, word, context_words):
        """Add or update word embedding"""
        word = word.lower()
        
        if word not in self.word_to_vec:
            self.word_to_vec[word] = np.random.randn(self.dim) * 0.01
            self.word_freq[word] = 0
        
        self.word_freq[word] += 1
        
        # Update embedding based on context
        for context_word in context_words:
            context_word = context_word.lower()
            if context_word in self.word_to_vec and context_word != word:
                # Move word vector closer to context
                self.word_to_vec[word] += 0.01 * (self.word_to_vec[context_word] - self.word_to_vec[word])
    
    def get_vector(self, word):
        """Get word vector"""
        word = word.lower()
        if word in self.word_to_vec:
            return self.word_to_vec[word]
        return np.zeros(self.dim)
    
    def similarity(self, word1, word2):
        """Calculate similarity between words"""
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def most_similar(self, word, n=5):
        """Find most similar words"""
        if word not in self.word_to_vec:
            return []
        
        similarities = []
        target_vec = self.word_to_vec[word]
        
        for other_word, other_vec in self.word_to_vec.items():
            if other_word != word:
                sim = np.dot(target_vec, other_vec) / (np.linalg.norm(target_vec) * np.linalg.norm(other_vec) + 1e-10)
                similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

class IntelligentKnowledge:
    """Advanced knowledge system with understanding"""
    
    def __init__(self):
        self.embeddings = WordEmbedding(dim=100)
        self.knowledge_graph = {}  # concept -> {related, facts, examples}
        self.learned_facts = []
        self.sentence_patterns = []
        self.articles_learned = 0
        self.is_learning = False
        self.waiting_for_topic = False
        self.last_unknown_query = ""
        
        # Built-in dictionary knowledge
        self.dictionary = self._build_dictionary()
        
    def _build_dictionary(self):
        """Basic dictionary with common words"""
        return {
            # Common words
            "hello": "a greeting or expression of goodwill",
            "computer": "an electronic device for storing and processing data",
            "learn": "to gain knowledge or skill by studying or experience",
            "artificial": "made by humans, not occurring naturally",
            "intelligence": "the ability to acquire and apply knowledge and skills",
            "neural": "relating to neurons or the nervous system",
            "network": "an interconnected system or group",
            "knowledge": "information and skills acquired through experience or education",
            "science": "systematic study of the natural world through observation and experiment",
            "wikipedia": "a free online encyclopedia",
        }
    
    def add_to_dictionary(self, word, definition):
        """Add word to dictionary"""
        self.dictionary[word.lower()] = definition
    
    def fetch_random_articles(self, count=10):
        """Fetch random Wikipedia articles"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'random',
                'rnnamespace': 0,
                'rnlimit': count
            }
            
            response = requests.get(url, params=params, timeout=10, 
                                   headers={'User-Agent': 'IntelligentBot/1.0'})
            data = response.json()
            
            return [page['title'] for page in data['query']['random']]
        except:
            return []
    
    def fetch_article(self, title):
        """Fetch article text"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'explaintext': True,
                'exsectionformat': 'plain'
            }
            
            response = requests.get(url, params=params, timeout=15,
                                   headers={'User-Agent': 'IntelligentBot/1.0'})
            data = response.json()
            
            pages = data['query']['pages']
            for page_id in pages:
                if 'extract' in pages[page_id]:
                    return pages[page_id]['extract']
            return None
        except:
            return None
    
    def extract_knowledge(self, text, topic):
        """Extract and internalize knowledge"""
        text = re.sub(r'\[\d+\]', '', text)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if 30 < len(s.strip()) < 300]
        
        for sentence in sentences[:40]:
            words = sentence.lower().split()
            
            # Build word embeddings
            for i, word in enumerate(words):
                if len(word) > 2:
                    context = words[max(0, i-3):i] + words[i+1:min(len(words), i+4)]
                    self.embeddings.add_word(word, context)
            
            # Extract definitions
            if ' is ' in sentence or ' are ' in sentence or ' means ' in sentence:
                self.learned_facts.append(sentence)
                
                # Try to extract subject and definition
                for keyword in [' is ', ' are ', ' means ']:
                    if keyword in sentence.lower():
                        parts = sentence.lower().split(keyword, 1)
                        if len(parts) == 2:
                            subject = parts[0].strip().split()[-3:]  # Last few words
                            subject = ' '.join(subject)
                            definition = parts[1].strip()
                            
                            if len(subject) < 30 and len(definition) < 200:
                                self.add_to_dictionary(subject, definition)
            
            # Store sentence patterns
            if len(self.sentence_patterns) < 1000:
                self.sentence_patterns.append(sentence)
        
        # Build knowledge graph
        if topic not in self.knowledge_graph:
            self.knowledge_graph[topic] = {
                'facts': [],
                'related': set(),
                'sentences': []
            }
        
        self.knowledge_graph[topic]['sentences'] = sentences[:20]
        self.knowledge_graph[topic]['facts'] = [s for s in sentences if ' is ' in s or ' are ' in s][:10]
    
    def learn_from_user(self, user_input, callback=None):
        """Learn from what the user teaches"""
        user_lower = user_input.lower()
        
        # Extract teaching patterns
        teaching_keywords = [' is ', ' are ', ' means ', ' refers to ', ' represents ', ' defines ']
        
        for keyword in teaching_keywords:
            if keyword in user_lower:
                parts = user_lower.split(keyword, 1)
                if len(parts) == 2:
                    subject = parts[0].strip()
                    definition = parts[1].strip()
                    
                    # Clean up the definition
                    definition = definition.rstrip('.!?')
                    
                    if 2 < len(subject) < 50 and 5 < len(definition) < 300:
                        self.add_to_dictionary(subject, definition)
                        self.learned_facts.append(user_input)
                        
                        # Also add to knowledge graph
                        if subject not in self.knowledge_graph:
                            self.knowledge_graph[subject] = {
                                'facts': [],
                                'related': set(),
                                'sentences': []
                            }
                        
                        self.knowledge_graph[subject]['facts'].append(user_input)
                        self.knowledge_graph[subject]['sentences'].append(user_input)
                        
                        return f"✅ Got it! I learned that {subject} {keyword.strip()} {definition}"
        
        return None
    
    def search_and_learn_topic(self, topic, callback=None):
        """Search Wikipedia for a specific topic and learn it"""
        if callback:
            callback(f"🔍 Searching Wikipedia for '{topic}'...")
        
        # First try exact match
        text = self.fetch_article(topic)
        
        if not text:
            # Try with capital first letter
            text = self.fetch_article(topic.capitalize())
        
        if not text:
            # Try searching
            if callback:
                callback(f"❌ Couldn't find '{topic}' on Wikipedia")
            return False
        
        if callback:
            callback(f"📖 Found article! Learning about {topic}...")
        
        self.extract_knowledge(text, topic)
        self.articles_learned += 1
        
        if callback:
            callback(f"✅ Successfully learned about {topic}!")
        
        return True
        """Learn from article"""
        if callback:
            callback(f"📚 Reading: {title}")
        
        text = self.fetch_article(title)
        if not text or len(text) < 100:
            if callback:
                callback(f"❌ Skipped")
            return False
        
        if callback:
            callback(f"✓ Processing {len(text)} chars")
        
        self.extract_knowledge(text, title)
        self.articles_learned += 1
        
        if callback:
            callback(f"✅ Learned! Total: {self.articles_learned} articles, {len(self.embeddings.word_to_vec)} words")
        
        return True
    
    def learn_article(self, title, callback=None):
        """Learn from article"""
        if callback:
            callback(f"📚 Reading: {title}")
        
        text = self.fetch_article(title)
        if not text or len(text) < 100:
            if callback:
                callback(f"❌ Skipped")
            return False
        
        if callback:
            callback(f"✓ Processing {len(text)} chars")
        
        self.extract_knowledge(text, title)
        self.articles_learned += 1
        
        if callback:
            callback(f"✅ Learned! Total: {self.articles_learned} articles, {len(self.embeddings.word_to_vec)} words")
        
        return True
    
    def generate_intelligent_response(self, user_input, callback=None):
        """Generate smart, contextual response"""
        
        # Check if we're waiting for a topic to learn
        if self.waiting_for_topic:
            self.waiting_for_topic = False
            topic = user_input.strip()
            
            if topic.lower() in ['skip', 'never mind', 'no', 'cancel', 'nah', 'stop']:
                self.last_unknown_query = ""
                return "Okay, maybe another time!"
            
            # Learn the topic
            if callback:
                callback("🔍 Searching Wikipedia...")
            
            success = self.search_and_learn_topic(topic, callback)
            
            if success:
                self.last_unknown_query = ""
                return f"✅ I just learned about '{topic}'! Now you can ask me questions about it."
            else:
                self.last_unknown_query = ""
                return f"❌ Sorry, I couldn't find '{topic}' on Wikipedia. Try:\n• A different topic\n• Teach me: '{topic} is ...'"
        
        # Try to learn from user's statement first
        learned = self.learn_from_user(user_input, callback)
        if learned:
            return learned
        
        user_lower = user_input.lower()
        
        # Extract key words (remove common question words and stopwords)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'who', 'when', 'where', 'why', 'how', 'do', 'does', 'did'}
        words = [w for w in user_lower.split() if len(w) > 2 and w not in stopwords]
        
        # Check for questions
        is_question = user_input.strip().endswith('?') or any(q in user_lower for q in ['what', 'why', 'how', 'when', 'where', 'who'])
        
        # Greetings
        if any(w in user_lower for w in ['hello', 'hi', 'hey', 'greetings']):
            return f"Hello! I've learned from {self.articles_learned} Wikipedia articles and understand {len(self.embeddings.word_to_vec)} words. Ask me anything!"
        
        # About self
        if any(phrase in user_lower for phrase in ['who are you', 'what are you', 'about yourself']):
            return (f"I'm an intelligent, self-learning AI. I've studied {self.articles_learned} Wikipedia articles "
                   f"and built semantic understanding of {len(self.embeddings.word_to_vec)} words. "
                   f"I learn from you too! If I don't know something, I'll search Wikipedia or you can teach me directly!")
        
        # Math
        if any(op in user_input for op in ['+', '-', '*', '/', '×', '÷']):
            try:
                expr = user_input.replace('×', '*').replace('÷', '/').replace('=', '').strip()
                # Remove any text before/after the expression
                expr = re.sub(r'[a-zA-Z]+', '', expr).strip()
                if expr:
                    result = eval(expr)
                    return f"The answer is {result}."
            except:
                pass
        
        # Search for knowledge in our database
        found_knowledge = []
        
        # Search knowledge graph by topic name
        for topic, data in self.knowledge_graph.items():
            topic_lower = topic.lower()
            
            # Check if any of the user's words match the topic
            for word in words:
                if word in topic_lower or topic_lower in user_lower:
                    if data['facts']:
                        found_knowledge.extend(data['facts'][:2])
                    elif data['sentences']:
                        found_knowledge.extend(data['sentences'][:2])
                    break
        
        # Search dictionary
        for word in words:
            if word in self.dictionary:
                found_knowledge.append(f"{word.capitalize()}: {self.dictionary[word]}")
        
        # If we found knowledge, return it
        if found_knowledge:
            if is_question:
                response = "Based on what I know:\n\n"
                unique = list(dict.fromkeys(found_knowledge))
                response += unique[0]
                
                if len(unique) > 1:
                    response += "\n\n" + unique[1]
                
                return response
            else:
                # Just making a statement - acknowledge it
                return f"Yes, I know about that! {found_knowledge[0]}"
        
        # No direct knowledge found - check semantic similarity
        if words:
            best_similarity = 0
            best_topic = None
            best_data = None
            
            for word in words[:2]:  # Check first 2 important words
                for topic, data in self.knowledge_graph.items():
                    similarity = self.embeddings.similarity(word, topic.lower())
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_topic = topic
                        best_data = data
            
            # Found something somewhat related
            if best_topic and best_similarity > 0.4:
                if best_data and best_data['sentences']:
                    return f"I don't know exactly, but here's something related about {best_topic}:\n\n{best_data['sentences'][0]}"
        
        # We don't know anything about this - ask for a topic to learn
        self.waiting_for_topic = True
        self.last_unknown_query = user_input
        
        # Extract a good topic suggestion from the user's input
        topic_suggestion = words[0] if words else None
        
        # Make the prompt clearer
        response = "🤔 I don't know about that yet!\n\n"
        
        if topic_suggestion:
            response += f"Give me a topic to search on Wikipedia (try '{topic_suggestion}'):\n"
        else:
            response += "Give me a topic to search on Wikipedia:\n"
        
        response += "• Just type the topic name\n"
        response += "• Or teach me: 'X is Y'\n"
        response += "• Or say 'skip' to cancel"
        
        return response

class LLMGUI:
    """GUI for the LLM"""
    
    def __init__(self):
        self.knowledge = IntelligentKnowledge()
        
        # Create window
        self.window = tk.Tk()
        self.window.title("Intelligent Self-Learning LLM")
        self.window.geometry("900x700")
        self.window.configure(bg='#1e1e1e')
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        self._create_widgets()
        self._load_knowledge()
    
    def _create_widgets(self):
        """Create GUI widgets"""
        # Top frame - stats
        top_frame = tk.Frame(self.window, bg='#2d2d2d', pady=10)
        top_frame.pack(fill='x')
        
        self.stats_label = tk.Label(top_frame, text="Articles: 0 | Words: 0 | Status: Idle", 
                                     bg='#2d2d2d', fg='#00ff00', font=('Consolas', 10))
        self.stats_label.pack()
        
        # Chat display
        chat_frame = tk.Frame(self.window, bg='#1e1e1e')
        chat_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            bg='#2d2d2d', 
            fg='#ffffff',
            font=('Consolas', 11),
            insertbackground='white'
        )
        self.chat_display.pack(fill='both', expand=True)
        self.chat_display.config(state='disabled')
        
        # Input frame
        input_frame = tk.Frame(self.window, bg='#1e1e1e')
        input_frame.pack(fill='x', padx=10, pady=5)
        
        self.input_field = tk.Entry(
            input_frame, 
            bg='#2d2d2d', 
            fg='#ffffff',
            font=('Consolas', 11),
            insertbackground='white'
        )
        self.input_field.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.input_field.bind('<Return>', lambda e: self.send_message())
        
        send_btn = tk.Button(
            input_frame, 
            text="Send", 
            command=self.send_message,
            bg='#0066cc',
            fg='white',
            font=('Consolas', 10, 'bold'),
            cursor='hand2'
        )
        send_btn.pack(side='left')
        
        # Control buttons
        control_frame = tk.Frame(self.window, bg='#1e1e1e')
        control_frame.pack(fill='x', padx=10, pady=5)
        
        self.learn_btn = tk.Button(
            control_frame,
            text="🧠 Start Learning",
            command=self.toggle_learning,
            bg='#00aa00',
            fg='white',
            font=('Consolas', 9, 'bold'),
            cursor='hand2'
        )
        self.learn_btn.pack(side='left', padx=2)
        
        tk.Button(
            control_frame,
            text="💾 Save",
            command=self.save_knowledge,
            bg='#0066cc',
            fg='white',
            font=('Consolas', 9, 'bold'),
            cursor='hand2'
        ).pack(side='left', padx=2)
        
        tk.Button(
            control_frame,
            text="📂 Load",
            command=self.load_knowledge_file,
            bg='#cc6600',
            fg='white',
            font=('Consolas', 9, 'bold'),
            cursor='hand2'
        ).pack(side='left', padx=2)
        
        tk.Button(
            control_frame,
            text="📊 Stats",
            command=self.show_stats,
            bg='#6600cc',
            fg='white',
            font=('Consolas', 9, 'bold'),
            cursor='hand2'
        ).pack(side='left', padx=2)
        
        tk.Button(
            control_frame,
            text="🗑️ Clear",
            command=self.clear_chat,
            bg='#cc0000',
            fg='white',
            font=('Consolas', 9, 'bold'),
            cursor='hand2'
        ).pack(side='left', padx=2)
        
        # Welcome message
        self.add_to_chat("AI", "Hello! I'm an intelligent self-learning LLM. Click 'Start Learning' to let me study Wikipedia, or just start chatting!")
    
    def add_to_chat(self, sender, message):
        """Add message to chat"""
        self.chat_display.config(state='normal')
        
        if sender == "You":
            self.chat_display.insert('end', f"\n{sender}: ", 'user')
            self.chat_display.tag_config('user', foreground='#00aaff', font=('Consolas', 11, 'bold'))
        else:
            self.chat_display.insert('end', f"\n{sender}: ", 'ai')
            self.chat_display.tag_config('ai', foreground='#00ff00', font=('Consolas', 11, 'bold'))
        
        self.chat_display.insert('end', f"{message}\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see('end')
    
    def send_message(self):
        """Send user message"""
        message = self.input_field.get().strip()
        if not message:
            return
        
        self.input_field.delete(0, 'end')
        self.add_to_chat("You", message)
        
        # Generate response (pass callback for learning updates)
        response = self.knowledge.generate_intelligent_response(message, self.update_stats_label)
        self.add_to_chat("AI", response)
        
        self.update_stats()
    
    def toggle_learning(self):
        """Toggle learning"""
        if not self.knowledge.is_learning:
            self.knowledge.is_learning = True
            self.learn_btn.config(text="🛑 Stop Learning", bg='#cc0000')
            self.update_stats_label("Learning...")
            
            thread = threading.Thread(target=self.learning_loop, daemon=True)
            thread.start()
        else:
            self.knowledge.is_learning = False
            self.learn_btn.config(text="🧠 Start Learning", bg='#00aa00')
            self.update_stats_label("Stopped")
    
    def learning_loop(self):
        """Background learning"""
        while self.knowledge.is_learning:
            titles = self.knowledge.fetch_random_articles(5)
            
            for title in titles:
                if not self.knowledge.is_learning:
                    break
                
                self.knowledge.learn_article(title, self.update_stats_label)
                self.update_stats()
                time.sleep(1)
            
            time.sleep(0.5)
    
    def update_stats_label(self, status):
        """Update status label"""
        self.stats_label.config(
            text=f"Articles: {self.knowledge.articles_learned} | "
                 f"Words: {len(self.knowledge.embeddings.word_to_vec)} | "
                 f"Status: {status}"
        )
    
    def update_stats(self):
        """Update stats"""
        status = "Learning..." if self.knowledge.is_learning else "Idle"
        self.update_stats_label(status)
    
    def save_knowledge(self):
        """Save knowledge to file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".brain",
            filetypes=[("Brain files", "*.brain"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                data = {
                    'embeddings': self.knowledge.embeddings,
                    'knowledge_graph': self.knowledge.knowledge_graph,
                    'dictionary': self.knowledge.dictionary,
                    'learned_facts': self.knowledge.learned_facts,
                    'sentence_patterns': self.knowledge.sentence_patterns,
                    'articles_learned': self.knowledge.articles_learned
                }
                
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                
                messagebox.showinfo("Success", f"Knowledge saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
    
    def load_knowledge_file(self):
        """Load knowledge from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Brain files", "*.brain"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.knowledge.embeddings = data['embeddings']
                self.knowledge.knowledge_graph = data['knowledge_graph']
                self.knowledge.dictionary = data['dictionary']
                self.knowledge.learned_facts = data['learned_facts']
                self.knowledge.sentence_patterns = data['sentence_patterns']
                self.knowledge.articles_learned = data['articles_learned']
                
                self.update_stats()
                messagebox.showinfo("Success", f"Knowledge loaded from {file_path}")
                self.add_to_chat("System", f"Loaded {self.knowledge.articles_learned} articles of knowledge!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")
    
    def _load_knowledge(self):
        """Auto-load default knowledge"""
        if os.path.exists('llm_brain.brain'):
            try:
                with open('llm_brain.brain', 'rb') as f:
                    data = pickle.load(f)
                
                self.knowledge.embeddings = data['embeddings']
                self.knowledge.knowledge_graph = data['knowledge_graph']
                self.knowledge.dictionary = data['dictionary']
                self.knowledge.learned_facts = data['learned_facts']
                self.knowledge.sentence_patterns = data['sentence_patterns']
                self.knowledge.articles_learned = data['articles_learned']
                
                self.update_stats()
                self.add_to_chat("System", f"Auto-loaded {self.knowledge.articles_learned} articles!")
            except:
                pass
    
    def show_stats(self):
        """Show detailed stats"""
        stats = (
            f"📊 DETAILED STATISTICS\n\n"
            f"Articles Learned: {self.knowledge.articles_learned}\n"
            f"Vocabulary Size: {len(self.knowledge.embeddings.word_to_vec)}\n"
            f"Dictionary Entries: {len(self.knowledge.dictionary)}\n"
            f"Knowledge Topics: {len(self.knowledge.knowledge_graph)}\n"
            f"Learned Facts: {len(self.knowledge.learned_facts)}\n"
            f"Sentence Patterns: {len(self.knowledge.sentence_patterns)}"
        )
        
        messagebox.showinfo("Statistics", stats)
    
    def clear_chat(self):
        """Clear chat display"""
        self.chat_display.config(state='normal')
        self.chat_display.delete(1.0, 'end')
        self.chat_display.config(state='disabled')
        self.add_to_chat("System", "Chat cleared!")
    
    def run(self):
        """Run the GUI"""
        # Auto-save on close
        def on_closing():
            if messagebox.askokcancel("Quit", "Save knowledge before quitting?"):
                data = {
                    'embeddings': self.knowledge.embeddings,
                    'knowledge_graph': self.knowledge.knowledge_graph,
                    'dictionary': self.knowledge.dictionary,
                    'learned_facts': self.knowledge.learned_facts,
                    'sentence_patterns': self.knowledge.sentence_patterns,
                    'articles_learned': self.knowledge.articles_learned
                }
                
                with open('llm_brain.brain', 'wb') as f:
                    pickle.dump(data, f)
            
            self.knowledge.is_learning = False
            self.window.destroy()
        
        self.window.protocol("WM_DELETE_WINDOW", on_closing)
        self.window.mainloop()

if __name__ == "__main__":
    app = LLMGUI()
    app.run()
