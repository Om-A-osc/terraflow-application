import { useCallback, useEffect, useRef, useState } from 'react';
import { searchVillages } from '../services/api';

/**
 * Village search box.
 *
 * Village names repeat constantly across India, so every suggestion shows its
 * tehsil, district and state, and results are biased towards wherever the map
 * is currently looking.  Keystrokes are debounced and the previous request is
 * aborted, so typing never queues up a backlog of searches.
 */
export default function VillageSearch({ mapCenter, onSelect, disabled }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const [error, setError] = useState(null);

  const abortRef = useRef(null);
  const boxRef = useRef(null);
  // Choosing a suggestion puts its name in the box, which would otherwise
  // re-trigger the search and reopen the list over the map.
  const justSelectedRef = useRef(false);
  // The map centre only biases ranking.  Keeping it in a ref rather than in the
  // effect's dependencies stops the fly-to that follows a selection from
  // re-running the search and popping the list back open over the map.
  const centerRef = useRef(mapCenter);
  useEffect(() => {
    centerRef.current = mapCenter;
  }, [mapCenter]);

  useEffect(() => {
    const text = query.trim();
    if (justSelectedRef.current) {
      justSelectedRef.current = false;
      return undefined;
    }
    if (text.length < 2) {
      // Clearing is handled where the text changes, so there is nothing to do
      return undefined;
    }

    const timer = setTimeout(async () => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setLoading(true);
      setError(null);
      try {
        const data = await searchVillages(text, centerRef.current, 8, controller.signal);
        setResults(data.results ?? []);
        setHighlight(0);
        setOpen(true);
      } catch (err) {
        if (err.name !== 'CanceledError' && err.name !== 'AbortError') {
          setError(err.message);
          setResults([]);
        }
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    const onClickOutside = (event) => {
      if (boxRef.current && !boxRef.current.contains(event.target)) setOpen(false);
    };
    document.addEventListener('mousedown', onClickOutside);
    return () => document.removeEventListener('mousedown', onClickOutside);
  }, []);

  const choose = useCallback(
    (village) => {
      justSelectedRef.current = true;
      abortRef.current?.abort();
      setQuery(village.name);
      setOpen(false);
      setResults([]);
      onSelect(village);
    },
    [onSelect],
  );

  const onKeyDown = (event) => {
    if (!open || results.length === 0) return;
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setHighlight((h) => Math.min(h + 1, results.length - 1));
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setHighlight((h) => Math.max(h - 1, 0));
    } else if (event.key === 'Enter') {
      event.preventDefault();
      choose(results[highlight]);
    } else if (event.key === 'Escape') {
      setOpen(false);
    }
  };

  return (
    <div className="village-search" ref={boxRef}>
      <div className="search-input-wrap">
        <span className="search-icon" aria-hidden="true">⌕</span>
        <input
          type="search"
          className="search-input"
          placeholder="Search any Indian village…"
          value={query}
          disabled={disabled}
          onChange={(e) => {
            const text = e.target.value;
            setQuery(text);
            if (text.trim().length < 2) {
              setResults([]);
              setOpen(false);
            }
          }}
          onFocus={() => results.length && setOpen(true)}
          onKeyDown={onKeyDown}
          aria-label="Search for a village"
          autoComplete="off"
        />
        {loading && <span className="search-spinner" aria-hidden="true" />}
      </div>

      {error && <div className="search-error">{error}</div>}

      {open && results.length > 0 && (
        <ul className="search-results" role="listbox">
          {results.map((village, index) => (
            <li
              key={village.id}
              role="option"
              aria-selected={index === highlight}
              className={`search-result ${index === highlight ? 'active' : ''}`}
              onMouseEnter={() => setHighlight(index)}
              onMouseDown={(e) => {
                e.preventDefault();
                choose(village);
              }}
            >
              <span className="result-name">{village.name}</span>
              <span className="result-where">
                {[village.subdistrict, village.district, village.state].filter(Boolean).join(' · ')}
              </span>
            </li>
          ))}
        </ul>
      )}

      {open && !loading && results.length === 0 && query.trim().length >= 2 && !error && (
        <div className="search-empty">
          No match. Try a different spelling, or pan the map and draw an area directly.
        </div>
      )}
    </div>
  );
}
