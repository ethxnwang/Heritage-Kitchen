"""Command-line entry point.

    python -m heritage_kitchen serve
    python -m heritage_kitchen suggest --culture Filipino --dish adobo --goal lower_sodium
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="heritage_kitchen")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    serve.add_argument("--reload", action="store_true")

    suggest = sub.add_parser("suggest", help="print substitutions as JSON")
    suggest.add_argument("--culture")
    suggest.add_argument("--cuisine")
    suggest.add_argument("--dish")
    suggest.add_argument("--ingredient", action="append", default=[], dest="ingredients")
    suggest.add_argument("--goal", action="append", default=[], dest="health_goals")
    suggest.add_argument("--restriction", action="append", default=[], dest="restrictions")
    suggest.add_argument("--no-recipe", action="store_true")
    suggest.add_argument("--offline", action="store_true", help="skip the LLM and use local ranking only")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("heritage_kitchen.api:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    if args.offline:
        os.environ["HK_OFFLINE"] = "1"

    from .engine import SubstitutionEngine
    from .models import SubstitutionRequest

    req = SubstitutionRequest(
        culture=args.culture,
        cuisine=args.cuisine,
        dish=args.dish,
        ingredients=args.ingredients,
        health_goals=args.health_goals,
        restrictions=args.restrictions,
        include_recipe=not args.no_recipe,
    )
    result = SubstitutionEngine().run(req)
    json.dump(result.model_dump(), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
