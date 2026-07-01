# Prompt Bibles

This folder is the locked reference context for the fruit cartoon prompt workflow.

Use direct lookups for locked Phase 3 and Phase 4 prompts:

```bash
python3 tools/prompt_generator.py character Appy
python3 tools/prompt_generator.py character AppyHero
python3 tools/prompt_generator.py environment VillageHomeExterior
```

Use deterministic bible assembly for Phase 5 and Phase 6 prompts:

```bash
python3 tools/prompt_generator.py scene --characters Appy,Ozzy --environment VillageHomeExterior --shot "close-up" --moment "Appy realizes Ozzy lied"
python3 tools/prompt_generator.py video --characters Appy,Ozzy --environment VillageHomeExterior --speaker1 Appy --line1 "Tumne mujhse jhoot kyun bola?" --emotion1 "hurt, quietly betrayed" --bgm "Betrayal reveal" --mood heartbreaking
```

Use Ollama only when you want local model judgment on top of the bibles:

```bash
ollama serve
ollama pull qwen2.5:14b
python3 tools/prompt_generator.py scene --use-ollama --characters Appy,Ozzy --environment VillageHomeExterior --shot "close-up" --moment "Appy realizes Ozzy lied"
OLLAMA_MODEL=llama3.1:8b python3 tools/prompt_generator.py scene --use-ollama --characters Appy,Ozzy --environment VillageHomeExterior --shot "close-up" --moment "Appy realizes Ozzy lied"
```
