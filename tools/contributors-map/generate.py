#!/usr/bin/env python3
"""Generate a contributor heatmap for the Fesod project.

This script fetches all unique contributors of a GitHub repository, resolves
profile locations to latitude/longitude coordinates, renders an ECharts-based
heatmap, and captures a static PNG screenshot that can be published without
committing artifacts to the repository.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from PIL import Image

API_BASE = "https://api.github.com"
DEFAULT_OUTPUT_DIR = Path("dist/contributor-heatmap")
CACHE_DIR = Path(os.environ.get("FESOD_CONTRIB_CACHE", Path.home() / ".cache/fesod-contributors"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "geocode-cache.json"
WORLD_GEOJSON_PATH = Path(__file__).with_name("world-geo.json")


class GitHubAPIError(RuntimeError):
    pass


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _load_cache() -> Dict[str, Dict[str, float]]:
    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except json.JSONDecodeError:
            _log("Warning: geocode cache is corrupted. Starting fresh.")
    return {}


def _save_cache(cache: Dict[str, Dict[str, float]]) -> None:
    with CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def paginate(url: str, headers: Dict[str, str]) -> Iterable[Dict[str, object]]:
    while url:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code >= 400:
            raise GitHubAPIError(
                f"GitHub API error {response.status_code} for {url}: {response.text}"
            )
        for item in response.json():
            yield item
        url = response.links.get("next", {}).get("url")


def _build_github_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def fetch_contributors(owner: str, repo: str, token: Optional[str]) -> List[str]:
    headers = _build_github_headers(token)
    url = f"{API_BASE}/repos/{owner}/{repo}/contributors?per_page=100"
    logins: List[str] = []
    for contributor in paginate(url, headers):
        login = contributor.get("login")
        ctype = contributor.get("type")
        if login and ctype == "User":
            logins.append(login)
    unique = sorted(set(logins))
    _log(f"Fetched {len(unique)} unique contributors")
    return unique


def fetch_user_profile(login: str, token: Optional[str]) -> Dict[str, Optional[str]]:
    headers = _build_github_headers(token)
    url = f"{API_BASE}/users/{login}"
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code >= 400:
        raise GitHubAPIError(
            f"GitHub API error {response.status_code} for {url}: {response.text}"
        )
    data = response.json()
    return {
        "login": data.get("login"),
        "name": data.get("name"),
        "location": data.get("location"),
        "avatar_url": data.get("avatar_url"),
        "html_url": data.get("html_url"),
    }


def geocode_locations(
    profiles: Iterable[Dict[str, Optional[str]]],
    delay_seconds: float = 1.0,
) -> List[Dict[str, object]]:
    cache = _load_cache()
    geolocator = Nominatim(user_agent="fesod-contributor-map")
    rate_limited_geocode = RateLimiter(geolocator.geocode, min_delay_seconds=delay_seconds)

    enriched: List[Dict[str, object]] = []
    for profile in profiles:
        location = profile.get("location")
        if location:
            cached = cache.get(location)
        else:
            cached = None
        lat_lon: Optional[Tuple[float, float]] = None
        if cached:
            lat_lon = (cached["lat"], cached["lon"])
            _log(f"Cache hit for '{location}' -> {lat_lon}")
        elif location:
            try:
                result = rate_limited_geocode(location, timeout=15)
            except Exception as exc:  # noqa: BLE001
                _log(f"Geocoding error for '{location}': {exc}")
                result = None
            if result:
                lat_lon = (result.latitude, result.longitude)
                cache[location] = {"lat": result.latitude, "lon": result.longitude}
                _log(f"Geocoded '{location}' -> {lat_lon}")
            else:
                _log(f"Unable to geocode '{location}'")
        enriched.append({
            **profile,
            "latitude": lat_lon[0] if lat_lon else None,
            "longitude": lat_lon[1] if lat_lon else None,
        })
    _save_cache(cache)
    return enriched


def _aggregate_points(
    contributors: List[Dict[str, object]],
    intensity_scale: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], float, float]:
    groups: Dict[Tuple[float, float], Dict[str, object]] = {}
    for entry in contributors:
        lat = entry.get("latitude")
        lon = entry.get("longitude")
        if lat is None or lon is None:
            continue
        key = (round(float(lon), 4), round(float(lat), 4))
        group = groups.setdefault(
            key,
            {
                "lon": float(lon),
                "lat": float(lat),
                "count": 0,
                "locations": Counter(),
                "logins": [],
            },
        )
        group["count"] += 1
        location_label = entry.get("location") or "Unknown"
        group["locations"][location_label] += 1
        if entry.get("login"):
            group["logins"].append(entry["login"])

    max_weight = 0.0
    min_weight = float("inf")
    data_points: List[Dict[str, object]] = []
    scatter_points: List[Dict[str, object]] = []
    for group in groups.values():
        weight = max(group["count"] * intensity_scale, 1.0)
        min_weight = min(min_weight, weight)
        if weight > max_weight:
            max_weight = weight
        dominant_location, _ = max(group["locations"].items(), key=lambda item: item[1])
        data_points.append(
            {
                "name": dominant_location,
                "value": [group["lon"], group["lat"], weight],
                "count": group["count"],
                "logins": group["logins"],
            }
        )
        scatter_points.append(
            {
                "name": dominant_location,
                "value": [group["lon"], group["lat"], group["count"]],
                "count": group["count"],
                "logins": group["logins"],
                "symbolSize": min(22, 8 + group["count"] * 4),
            }
        )

    if min_weight == float("inf"):
        min_weight = 1.0

    return data_points, scatter_points, min_weight, max_weight


def build_heatmap(
    contributors: List[Dict[str, object]],
    output_dir: Path,
    map_title: str,
    width: int,
    height: int,
    capture_png: bool = True,
    world_geojson_path: Path = WORLD_GEOJSON_PATH,
    intensity_scale: float = 6.0,
) -> Path:
    data_points, scatter_points, min_weight, max_weight = _aggregate_points(
        contributors, intensity_scale
    )

    if not data_points:
        raise RuntimeError("No contributor locations available to plot.")

    geojson_obj = json.loads(world_geojson_path.read_text(encoding="utf-8"))
    max_weight = max(max_weight, intensity_scale)
    min_weight = min(min_weight, max_weight)

    tooltip_formatter = (
        "function (params) {\n"
        "                const names = params.data.logins || [];\n"
        "                const list = names.length ? names.join(', ') : 'No public GitHub location';\n"
        "                const count = params.data.count || params.data.value[2];\n"
        "                return `${params.name}<br/>Contributors: ${count}<br/>${list}`;\n"
        "            }"
    )

    option = {
        "backgroundColor": "#f9fafb",
        "title": {
            "text": map_title,
            "left": "center",
            "top": 10,
            "textStyle": {"color": "#1f2937", "fontSize": 20},
        },
        "tooltip": {
            "trigger": "item",
            "formatter": tooltip_formatter,
        },
        "visualMap": {
            "min": max(1.0, min_weight),
            "max": max_weight,
            "calculable": True,
            "inRange": {"color": ["#fef3c7", "#f97316", "#b91c1c"]},
            "textStyle": {"color": "#111827"},
            "left": 20,
            "bottom": 20,
        },
        "geo": {
            "map": "custom_world",
            "roam": True,
            "silent": False,
            "zoom": 1,
            "scaleLimit": {"min": 1, "max": 10},
            "itemStyle": {
                "areaColor": "#f3f4f6",
                "borderColor": "#d1d5db",
            },
            "emphasis": {
                "itemStyle": {"areaColor": "#bfdbfe"},
                "label": {"color": "#1f2937"},
            },
        },
        "series": [
            {
                "name": "Contributors",
                "type": "heatmap",
                "coordinateSystem": "geo",
                "data": data_points,
                "pointSize": 18,
                "blurSize": 28,
            },
            {
                "name": "Contributor Locations",
                "type": "effectScatter",
                "coordinateSystem": "geo",
                "data": scatter_points,
                "tooltip": {"trigger": "item"},
                "showEffectOn": "render",
                "rippleEffect": {"scale": 4, "brushType": "stroke"},
                "itemStyle": {"color": "#dc2626", "shadowBlur": 10, "shadowColor": "rgba(220,38,38,0.5)"},
                "label": {"show": False, "formatter": "{b}", "position": "right", "color": "#1f2937"},
                "zlevel": 1,
            },
        ],
    }

    html_template = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta http-equiv=\"X-UA-Compatible\" content=\"IE=edge\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
        <title>{map_title}</title>
        <style>
            html, body {{
                height: 100%;
                margin: 0;
                background: #f3f4f6;
                color: #1f2937;
                font-family: 'Segoe UI', Roboto, sans-serif;
            }}
            #chart {{
                width: 100%;
                height: 100%;
            }}
        </style>
        <script src=\"https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js\"></script>
    </head>
    <body>
        <div id=\"chart\"></div>
        <script>
            const geoJson = {json.dumps(geojson_obj, ensure_ascii=False)};
            echarts.registerMap('custom_world', geoJson);
            const chart = echarts.init(document.getElementById('chart'));
            const option = {json.dumps(option, ensure_ascii=False)};
            option.tooltip.formatter = {tooltip_formatter};
            chart.setOption(option);
            window.addEventListener('resize', () => chart.resize());
            chart.on('finished', () => {{ window.__chartReady = true; }});
        </script>
    </body>
    </html>
    """

    html_path = output_dir / "contributors-heatmap.html"
    html_path.write_text(html_template, encoding="utf-8")

    if capture_png:
        from playwright.sync_api import sync_playwright  # Imported lazily to allow --skip-screenshot

        png_path = output_dir / "contributors-heatmap.png"
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport={"width": width, "height": height},
                device_scale_factor=2,
            )
            page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            page.wait_for_function("window.__chartReady === true", timeout=20000)
            page.wait_for_timeout(1000)
            page.screenshot(path=str(png_path), full_page=True)
            browser.close()

        with Image.open(png_path) as img:
            bbox = img.getbbox()
            if bbox:
                img.crop(bbox).save(png_path)

    return html_path


def summarize(contributors: List[Dict[str, object]]) -> str:
    total = len(contributors)
    located = sum(1 for c in contributors if c.get("latitude") is not None)
    return textwrap.dedent(
        f"""
        Total contributors analysed: {total}
        Contributors with location: {located}
        """
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=os.environ.get("GITHUB_REPOSITORY_OWNER", "alaahong"))
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "fesod").split("/")[-1])
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub token")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=1600, help="Screenshot width in pixels")
    parser.add_argument("--height", type=int, default=900, help="Screenshot height in pixels")
    parser.add_argument("--map-title", default="Fesod Contributor Heatmap")
    parser.add_argument(
        "--sample", action="store_true", help="Generate a sample map with synthetic data for testing"
    )
    parser.add_argument(
        "--skip-screenshot", action="store_true", help="Skip PNG capture (useful for headless tests)"
    )
    parser.add_argument(
        "--world-geojson",
        type=Path,
        default=WORLD_GEOJSON_PATH,
        help="Path to the world GeoJSON file used for rendering (defaults to bundled dataset)",
    )
    parser.add_argument(
        "--intensity-scale",
        type=float,
        default=6.0,
        help="Multiplier applied to each location's contributor count to strengthen heat intensity",
    )
    return parser.parse_args()


def build_sample_dataset() -> List[Dict[str, object]]:
    sample_points = [
        {
            "login": "alice",
            "name": "Alice",
            "location": "San Francisco, USA",
            "latitude": 37.7749,
            "longitude": -122.4194,
        },
        {
            "login": "bob",
            "name": "Bob",
            "location": "Berlin, Germany",
            "latitude": 52.52,
            "longitude": 13.4050,
        },
        {
            "login": "carla",
            "name": "Carla",
            "location": "Sydney, Australia",
            "latitude": -33.8688,
            "longitude": 151.2093,
        },
        {
            "login": "dan",
            "name": "Dan",
            "location": "Tokyo, Japan",
            "latitude": 35.6762,
            "longitude": 139.6503,
        },
        {
            "login": "erin",
            "name": "Erin",
            "location": "São Paulo, Brazil",
            "latitude": -23.5558,
            "longitude": -46.6396,
        },
    ]
    results: List[Dict[str, object]] = []
    for entry in sample_points:
        results.append(
            {
                "login": entry["login"],
                "name": entry["name"],
                "location": entry["location"],
                "latitude": entry["latitude"],
                "longitude": entry["longitude"],
                "avatar_url": None,
                "html_url": None,
            }
        )
    return results


def main() -> None:
    args = parse_args()

    if not args.token and not args.sample:
        _log(
            "Warning: running without a GitHub token. API rate limits are lower and the run may fail if"
            " too many requests are required."
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample:
        _log("Generating sample dataset...")
        contributors = build_sample_dataset()
    else:
        logins = fetch_contributors(args.owner, args.repo, args.token)
        profiles = [fetch_user_profile(login, args.token) for login in logins]
        contributors = geocode_locations(profiles)

    located_contributors = [c for c in contributors if c.get("latitude") is not None]
    if not located_contributors:
        raise SystemExit("No contributor locations resolved; cannot create heatmap.")

    if not args.world_geojson.exists():
        raise SystemExit(f"World GeoJSON not found at {args.world_geojson}. Please provide a valid file via --world-geojson.")

    html_path = build_heatmap(
        located_contributors,
        output_dir=output_dir,
        map_title=args.map_title,
        width=args.width,
        height=args.height,
        capture_png=not args.skip_screenshot,
        world_geojson_path=args.world_geojson,
        intensity_scale=max(args.intensity_scale, 1.0),
    )

    data_path = output_dir / "contributors.json"
    with data_path.open("w", encoding="utf-8") as f:
        json.dump(contributors, f, indent=2)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summarize(contributors), encoding="utf-8")

    _log(f"Heatmap created at {output_dir}")
    _log(html_path.read_text(encoding="utf-8")[:200])


if __name__ == "__main__":
    main()
