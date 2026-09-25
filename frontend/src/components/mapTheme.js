/**
 * One palette for the whole map, with one hue per meaning.
 *
 * Overlays sit on satellite imagery, which is busy, mid-toned and mostly
 * green-brown.  Two rules keep them readable:
 *
 *   1. A colour means one thing.  Earlier the catchment, the drainage lines
 *      and the first pond site were all cyan, so three unrelated outlines
 *      looked like one broken outline.
 *   2. Every bright line gets a dark casing drawn underneath it.  A 1 px halo
 *      is what makes a thin line legible over imagery, and it is why map
 *      styles from OS, Google and Mapbox all use one.
 *
 * Dashes carry meaning too, so they are rationed: solid is something measured,
 * dashed is something proposed or approximate.
 */

export const MAP = {
  // The area the user drew.  White reads on any imagery and belongs to no
  // other layer, so the selection never competes with a result.
  selection: { color: '#ffffff', weight: 2.5, fill: false, dash: null },

  // Catchments are boundaries, not water, so they are deliberately not blue.
  // Drawn in sky blue they were mistaken for the drainage lines running under
  // them, which is the one confusion this palette most needs to avoid.
  catchment: { color: '#f472b6', weight: 2.5, fillOpacity: 0.06, dash: null },
  // Dashed when it runs off the edge of the analysed window: the boundary is
  // then a lower bound, not a measurement.
  catchmentTruncated: { color: '#f472b6', weight: 2.5, fillOpacity: 0.06, dash: '8,6' },

  // The catchment of one chosen pond: the same family, one step lighter.
  siteCatchment: { color: '#f9a8d4', weight: 2, fillOpacity: 0.05, dash: '6,5' },

  // Natural drainage.  The only blue on the map, because it is the only water.
  stream: { color: '#22d3ee' },

  // Where water would stand.
  pond: { color: '#34d399', weight: 2.5, fillOpacity: 0.5 },
  pondMuted: { color: '#34d399', weight: 1.5, fillOpacity: 0.15 },

  // Proposed works: dashed, because none of it exists yet.
  afterWorks: { color: '#fbbf24', weight: 2.5, fillOpacity: 0.28, dash: '7,5' },
  bund: { color: '#f97316', weight: 3, fillOpacity: 0.55 },
  haul: { color: '#fde68a', weight: 1.5 },

  // Ground a pond cannot go on.
  constraintWater: { color: '#3b82f6', fillOpacity: 0.28 },
  constraintForest: { color: '#22c55e', fillOpacity: 0.22 },
  constraintBuilt: { color: '#f87171', fillOpacity: 0.22 },

  // Context only, so it stays quiet.
  village: { color: '#e2e8f0', weight: 1.5, dash: '2,6' },

  // The halo under every bright line.
  casing: '#0b1220',
};

/** Width of a drainage line for its Strahler order. */
export const streamWeight = (order) => Math.min(4.5, 1.1 + (order ?? 1) * 0.7);

/**
 * A bright line plus the dark line drawn underneath it.
 *
 * Leaflet has no casing option, so the layer is rendered twice and this
 * returns both styles.
 */
export function casedStyle({ color, weight, fillOpacity = 0, dash = null }) {
  return {
    casing: {
      color: MAP.casing,
      weight: weight + 2.5,
      opacity: 0.55,
      fill: false,
      dashArray: dash,
      lineCap: 'round',
      lineJoin: 'round',
      interactive: false,
    },
    line: {
      color,
      weight,
      opacity: 1,
      fillColor: color,
      fillOpacity,
      fill: fillOpacity > 0,
      dashArray: dash,
      lineCap: 'round',
      lineJoin: 'round',
    },
  };
}

/**
 * Sites are told apart by their rank number, not by colour.
 *
 * Giving each site its own hue meant the palette collided with the layer
 * colours and a reader had to consult the legend to tell a pond from a
 * catchment.  A number needs no legend.
 */
export const SITE = '#34d399';
export const SITE_SELECTED = '#fbbf24';

export const siteColor = (_index, selected = false) => (selected ? SITE_SELECTED : SITE);


/**
 * A dark sheet the size of the current view, with the selection punched out.
 *
 * The sheet is sized to the viewport rather than to the world.  A
 * world-covering polygon dims everything outside the selection at every zoom,
 * so zooming out turned the whole map black with a pinhole where the area was,
 * and its coordinates were large enough to risk rendering artefacts as well.
 *
 * Returns null when the selection is too small on screen to be worth framing,
 * which is the other half of the same problem: below that size the sheet is
 * all a reader would see.
 */
export function focusMask(geometry, viewBounds, coverage = 1) {
  if (!geometry || !viewBounds) return null;
  if (coverage < 0.015) return null;

  const pad = 0.4;
  const west = viewBounds.getWest();
  const east = viewBounds.getEast();
  const south = viewBounds.getSouth();
  const north = viewBounds.getNorth();
  const dx = (east - west) * pad;
  const dy = (north - south) * pad;

  const sheet = [
    [west - dx, south - dy],
    [east + dx, south - dy],
    [east + dx, north + dy],
    [west - dx, north + dy],
    [west - dx, south - dy],
  ];

  let holes = [];
  if (geometry.type === 'Polygon') {
    holes = [geometry.coordinates[0]];
  } else if (geometry.type === 'MultiPolygon') {
    holes = geometry.coordinates.map((poly) => poly[0]);
  }
  if (holes.length === 0) return null;

  return {
    type: 'Feature',
    properties: { kind: 'focus-mask' },
    geometry: { type: 'Polygon', coordinates: [sheet, ...holes] },
  };
}

export const FOCUS_MASK_STYLE = {
  stroke: false,
  fillColor: '#04070f',
  fillOpacity: 0.45,
  interactive: false,
};
