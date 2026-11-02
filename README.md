# Heritage Kitchen

Backend for [Heritage Kitchen](http://heritagekitchencabinets.com/), a site that suggests healthier ingredient swaps for the food people grew up with. You give it a culture, a cuisine, a specific dish, or a list of ingredients. It sends back substitutions that keep the dish recognizable, each with ratios, flavor and cultural notes, and nutrition numbers.

The front end is a single embeddable widget (`squarespace/embed.html`) that runs inside a Squarespace Code Block and calls this API.

## How it works

```
request ──► resolve cuisine ──► pick target ingredients ──► NumPy ranking ──► Claude ──► attach nutrition ──► response
            (culture / cuisine     (user list, or the         (health score,      (structured
             / dish lookup)         cuisine's heaviest         flavor cosine,      JSON output)
                                    staples)                   cultural fit)
```

1. **Cuisine resolution.** 34 cuisines live in `heritage_kitchen/data/cuisines.json`, each with a flavor fingerprint, pantry staples, the ingredients that usually carry the most fat/salt/sugar, signature dishes, and the culture names people actually type ("Punjabi", "Boricua", "Filipino-American"). A dish name alone is enough: "jollof rice" resolves to West African.
2. **Candidate ranking (NumPy).** `nutrition.py` scores every ingredient's per-100 g vector `(kcal, sat fat, sodium, sugar, fiber, protein)` against FDA daily values, with weights that shift by health goal. Candidates are ranked by health gain, cosine similarity between flavor vectors (blended with the user's taste sliders), and whether the substitute is already a staple in that cuisine. Dietary restrictions are hard filters.
3. **LLM step.** The ranked candidates, cuisine context, and user profile go to Claude (`claude-opus-5-5`, adaptive thinking, JSON-schema structured output). The system prompt tells the model to prefer swaps that already exist inside the tradition, protect the dish's flavor anchors, and treat the candidate list as evidence rather than orders. See `heritage_kitchen/prompts.py`.
4. **Grounding.** The model isn't trusted for numbers. After it responds, each `original → substitute` pair is matched back to the ingredient table and real nutrient deltas are attached.
5. **Fallback.** With no API key, or if the API call fails, the engine returns the top NumPy-ranked swap per ingredient with templated notes, so the site never shows an empty result.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # add your ANTHROPIC_API_KEY
export $(grep -v '^#' .env | xargs)
python -m heritage_kitchen serve --reload
```

Interactive docs at http://localhost:8000/docs.

From the command line:

```bash
python -m heritage_kitchen suggest --culture "Filipino-American" --dish "chicken adobo" --goal lower_sodium
python -m heritage_kitchen suggest --dish "butter chicken" --ingredient ghee --ingredient "heavy cream" --restriction vegetarian --offline
```

Tests run without network access:

```bash
pytest
```

## API

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/api/substitutions` | Main endpoint. Returns swaps, an adapted recipe, and tips. |
| `POST` | `/api/taste-match` | Ranks cuisines by how closely they match a taste profile. |
| `GET` | `/api/cuisines` | All supported cuisines (used to fill the dropdown). |
| `GET` | `/api/cuisines/{key}` | Staples, heavy ingredients, dishes, and flavor profile for one cuisine. |
| `GET` | `/api/ingredients?q=` | Nutrition lookup for one ingredient. |
| `GET` | `/api/health` | Liveness, plus whether the LLM is configured. |

Example request:

```json
POST /api/substitutions
{
  "culture": "Filipino-American",
  "dish": "chicken adobo",
  "ingredients": ["pork belly", "soy sauce", "white rice"],
  "taste": {"spicy": 3, "sweet": 4, "salty": 6, "sour": 9, "umami": 7, "bitter": 2},
  "health_goals": ["lower_sodium", "heart_health"],
  "restrictions": [],
  "include_recipe": true
}
```

Abbreviated response:

```json
{
  "cuisine": {"key": "filipino", "name": "Filipino", "region": "Southeast Asia"},
  "dish": "Chicken adobo",
  "source": "llm",
  "model": "claude-opus-5-5",
  "summary": "...",
  "substitutions": [
    {
      "original": "soy sauce",
      "substitute": "low-sodium soy sauce",
      "ratio": "1:1, add a splash more cane vinegar",
      "reason": "...",
      "flavor_note": "...",
      "cultural_note": "...",
      "nutrition": {
        "basis": "per 100 g",
        "original": {"kcal": 53.0, "sodium_mg": 5493.0, "...": "..."},
        "substitute": {"kcal": 53.0, "sodium_mg": 3333.0, "...": "..."},
        "change_pct": {"sodium_mg": -39.3, "...": "..."},
        "health_score_delta": 3.943
      }
    }
  ],
  "adapted_recipe": {"title": "...", "servings": 4, "ingredients": ["..."], "steps": ["..."]},
  "tips": ["..."]
}
```

`source` is `"offline"` when the response came from the local ranker instead of the model.

**Health goals:** `lower_sodium`, `heart_health`, `weight_management`, `blood_sugar`, `high_protein`, `more_fiber`

**Restrictions:** `vegetarian`, `vegan`, `pescatarian`, `dairy_free`, `gluten_free`, `halal`, `kosher`, `nut_free`, `shellfish_free`, `soy_free`

## Putting it on Squarespace

Squarespace only hosts the front end, so the API runs somewhere else and the page calls it.

1. **Deploy the backend.** The repo includes a `Dockerfile`, a `render.yaml` blueprint, and a `Procfile`, so Render, Railway, Fly.io, or Heroku all work. On Render: New → Blueprint → pick this repo, then set `ANTHROPIC_API_KEY` in the dashboard.
2. **Set CORS.** `HK_ALLOWED_ORIGINS` defaults to `heritagekitchencabinets.com` (with and without `www`), and `*.squarespace.com` is allowed so the editor preview works. Change it if the domain changes.
3. **Embed the widget.** Open `squarespace/embed.html`, set `HK_API_BASE` to your deployed URL, and paste the whole file into a Code Block on the page (Code Blocks with JavaScript need a Business plan or higher).

The widget's styles are scoped under `#hk-app`, so they won't clash with the site template.

## Project layout

```
heritage_kitchen/
  api.py          FastAPI app, CORS, rate limiting
  engine.py       request -> response pipeline and offline fallback
  nutrition.py    NumPy health scoring, flavor similarity, substitute ranking
  catalog.py      data loading and fuzzy lookup for cuisines, cultures, dishes, ingredients
  prompts.py      system prompt, user prompt builder, output JSON schema
  llm.py          Anthropic client wrapper
  models.py       request/response schemas
  data/           cuisines.json (34 cuisines), ingredients.json (98 ingredients)
squarespace/
  embed.html      drop-in front end for a Squarespace Code Block
tests/
```

## Notes on the data

Nutrient values are approximate per-100 g figures from USDA FoodData Central, rounded. They're meant for comparing one ingredient with another, not for tracking intake. Nothing here is medical advice.

MIT
