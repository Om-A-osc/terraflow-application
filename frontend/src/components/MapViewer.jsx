import { useEffect, useMemo, useRef, useState } from 'react';
import {
  GeoJSON, LayersControl, MapContainer, Marker, Polyline, Popup, ScaleControl,
  TileLayer, useMap,
} from 'react-leaflet';
import L from 'leaflet';
import DrawControl from './DrawControl';
import { FOCUS_MASK_STYLE, MAP, casedStyle, focusMask, streamWeight } from './mapTheme';

// Leaflet's default marker images do not survive bundling
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

// Free basemaps, no key required.  Attribution stays visible and tiles are
// never prefetched or proxied, which the OpenStreetMap policy asks for.
//
// maxNativeZoom is the setting that matters here: rural India has satellite
// imagery only to about zoom 18, and asking Esri for zoom 19 returns a grey
// "Map data not yet available" tile.  Capping the native zoom makes Leaflet
// upscale the last real tile instead, so zooming further keeps working.
const BASEMAPS = [
  {
    name: 'Satellite',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri, Maxar, Earthstar Geographics',
    maxNativeZoom: 18,
    checked: true,
  },
  {
    name: 'Streets',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxNativeZoom: 19,
  },
  {
    name: 'Terrain',
    url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)',
    maxNativeZoom: 17,
  },
];

// Two levels past the imagery's native resolution.  Beyond that the upscaled
// tile is too soft to place anything on, and a blurry map invites more trust
// than it deserves.
const MAX_ZOOM = 20;

/** A numbered pin, so sites are told apart without spending a colour on each. */
function siteIcon(rank, selected) {
  return L.divIcon({
    className: 'site-pin-wrap',
    html: `<span class="site-pin${selected ? ' is-selected' : ''}">${rank}</span>`,
    iconSize: selected ? [30, 30] : [24, 24],
    iconAnchor: selected ? [15, 15] : [12, 12],
  });
}

/** A bright outline over a dark casing, so a thin line reads on imagery. */
function CasedGeoJSON({ id, data, spec }) {
  const { casing, line } = casedStyle(spec);
  return (
    <>
      <GeoJSON key={`${id}-casing`} data={data} style={casing} />
      <GeoJSON key={id} data={data} style={line} />
    </>
  );
}

/**
 * Dims everything outside the selected area.
 *
 * The sheet follows the viewport and is rebuilt whenever the map moves, and it
 * removes itself once the selection is a small enough part of the view that
 * dimming would just black the screen.
 */
function FocusMask({ geometry, id }) {
  const map = useMap();
  const [mask, setMask] = useState(null);

  useEffect(() => {
    const rebuild = () => {
      try {
        const selBounds = L.geoJSON(geometry).getBounds();
        if (!selBounds.isValid()) {
          setMask(null);
          return;
        }
        // How much of the view the selection takes up, by screen area
        const nw = map.latLngToContainerPoint(selBounds.getNorthWest());
        const se = map.latLngToContainerPoint(selBounds.getSouthEast());
        const size = map.getSize();
        const coverage =
          (Math.abs(se.x - nw.x) * Math.abs(se.y - nw.y)) / Math.max(1, size.x * size.y);
        setMask(focusMask(geometry, map.getBounds(), coverage));
      } catch {
        setMask(null);
      }
    };

    rebuild();
    map.on('moveend zoomend resize', rebuild);
    return () => map.off('moveend zoomend resize', rebuild);
  }, [map, geometry]);

  if (!mask) return null;
  return <GeoJSON key={`focus-${id}-${JSON.stringify(mask.geometry.coordinates[0][0])}`}
    data={mask} style={FOCUS_MASK_STYLE} />;
}

function FlyTo({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (center) map.flyTo([center.lat, center.lon], zoom ?? 15, { duration: 1.1 });
  }, [center, zoom, map]);
  return null;
}

function FitTo({ geometry }) {
  const map = useMap();
  useEffect(() => {
    if (!geometry) return;
    try {
      const bounds = L.geoJSON(geometry).getBounds();
      if (bounds.isValid()) map.fitBounds(bounds, { padding: [48, 48], maxZoom: 17 });
    } catch {
      /* geometry not renderable */
    }
  }, [geometry, map]);
  return null;
}

/** Brings the chosen pond into view when it is picked from the list. */
function RevealSite({ site }) {
  const map = useMap();
  useEffect(() => {
    if (!site) return;
    const target = L.latLng(site.location.lat, site.location.lon);
    if (map.getBounds().pad(-0.15).contains(target)) return;   // already on screen
    map.panTo(target, { animate: true, duration: 0.6 });
  }, [site, map]);
  return null;
}

function TrackView({ onMove, onZoom }) {
  const map = useMap();
  useEffect(() => {
    const emit = () => {
      onMove?.(map.getCenter());
      onZoom?.(map.getZoom());
    };
    emit();
    map.on('moveend', emit);
    return () => map.off('moveend', emit);
  }, [map, onMove, onZoom]);
  return null;
}

export default function MapViewer({
  analysis,
  earthwork,
  selectedSiteId,
  onSelectSite,
  onGeometryChange,
  onInvalidGeometry,
  onMapMove,
  flyTo,
  villageBoundary,
  seedGeometry,
  layers,
  onToggleLayer,
}) {
  const mapRef = useRef(null);
  const [zoom, setZoom] = useState(5);

  const sites = analysis?.sites ?? [];
  const selectedIndex = sites.findIndex((s) => s.site_id === selectedSiteId);
  const selected = selectedIndex >= 0 ? sites[selectedIndex] : null;

  const catchmentSpec = useMemo(
    () => (analysis?.catchment?.truncated ? MAP.catchmentTruncated : MAP.catchment),
    [analysis?.catchment?.truncated],
  );

  return (
    <>
      <MapContainer
        center={[22.0, 79.0]}
        zoom={5}
        maxZoom={MAX_ZOOM}
        style={{ width: '100%', height: '100%' }}
        ref={mapRef}
        preferCanvas
      >
        <LayersControl position="topright">
          {BASEMAPS.map((base) => (
            <LayersControl.BaseLayer key={base.name} name={base.name} checked={base.checked}>
              <TileLayer
                url={base.url}
                attribution={base.attribution}
                maxNativeZoom={base.maxNativeZoom}
                maxZoom={MAX_ZOOM}
              />
            </LayersControl.BaseLayer>
          ))}
        </LayersControl>

        <ScaleControl position="bottomright" imperial={false} />
        <TrackView onMove={onMapMove} onZoom={setZoom} />
        <FlyTo center={flyTo} />
        <RevealSite site={selected} />
        <FitTo geometry={analysis?.selection?.geometry} />
        <DrawControl
          onChange={onGeometryChange}
          onInvalid={onInvalidGeometry}
          externalGeometry={seedGeometry}
        />

        {/* Drawn before everything else so it dims only the imagery */}
        {layers.focus && analysis?.selection?.geometry && (
          <FocusMask geometry={analysis.selection.geometry} id={analysis.analysis_id} />
        )}

        {/* Context: the village, quietly */}
        {layers.village && villageBoundary && (
          <GeoJSON
            key={`village-${JSON.stringify(villageBoundary).length}`}
            data={villageBoundary}
            style={{
              color: MAP.village.color,
              weight: MAP.village.weight,
              dashArray: MAP.village.dash,
              opacity: 0.55,
              fill: false,
              interactive: false,
            }}
          />
        )}

        {/* Ground a pond cannot go on, underneath everything else */}
        {layers.constraints && analysis?.constraints_geojson?.features?.length > 0 && (
          <GeoJSON
            key={`constraints-${analysis.analysis_id}`}
            data={analysis.constraints_geojson}
            style={(feature) => {
              const kind = feature?.properties?.kind;
              const spec =
                kind === 'water' ? MAP.constraintWater
                  : kind === 'forest' ? MAP.constraintForest
                    : MAP.constraintBuilt;
              return {
                color: spec.color,
                weight: 0.8,
                opacity: 0.75,
                fillColor: spec.color,
                fillOpacity: spec.fillOpacity,
                interactive: false,
              };
            }}
          />
        )}

        {/* What drains into the selected area */}
        {layers.catchment && analysis?.catchment?.feature && (
          <CasedGeoJSON
            id={`catchment-${analysis.analysis_id}`}
            data={analysis.catchment.feature}
            spec={catchmentSpec}
          />
        )}

        {/* What drains into the chosen pond: a different question, so a
            different colour, and only ever one of them on screen */}
        {layers.siteCatchment && selected?.catchment?.feature && (
          <CasedGeoJSON
            id={`site-catchment-${selected.site_id}`}
            data={selected.catchment.feature}
            spec={MAP.siteCatchment}
          />
        )}

        {/* Natural drainage */}
        {layers.streams && analysis?.streams_geojson?.features?.length > 0 && (
          <>
            <GeoJSON
              key={`streams-casing-${analysis.analysis_id}`}
              data={analysis.streams_geojson}
              style={(feature) => ({
                color: MAP.casing,
                weight: streamWeight(feature?.properties?.strahler_order) + 2,
                opacity: 0.5,
                fill: false,
                lineCap: 'round',
                interactive: false,
              })}
            />
            <GeoJSON
              key={`streams-${analysis.analysis_id}`}
              data={analysis.streams_geojson}
              style={(feature) => ({
                color: MAP.stream.color,
                weight: streamWeight(feature?.properties?.strahler_order),
                opacity: 0.95,
                fill: false,
                lineCap: 'round',
                lineJoin: 'round',
              })}
            />
          </>
        )}

        {/* Only the chosen pond is drawn.  Showing every footprint at once
            put several overlapping green shapes on the map and none of them
            could be told apart; the numbered pins carry the other options
            until one is picked. */}
        {layers.sites && selected?.footprint && (
          <GeoJSON
            key={`footprint-${selected.site_id}`}
            data={selected.footprint}
            style={{
              color: '#ffffff',
              weight: MAP.pond.weight,
              fillColor: MAP.pond.color,
              fillOpacity: MAP.pond.fillOpacity,
            }}
          />
        )}

        {/* The pond after the proposed works */}
        {layers.sites && earthwork?.new_footprint && (
          <CasedGeoJSON
            id={`after-works-${earthwork.site_id}-${earthwork.storage?.new_capacity_m3}`}
            data={earthwork.new_footprint}
            spec={MAP.afterWorks}
          />
        )}

        {earthwork?.bund_geometry && (
          <GeoJSON
            key={`bund-${earthwork.site_id}-${earthwork.design?.spill_raise_m}`}
            data={earthwork.bund_geometry}
            style={{
              color: MAP.bund.color,
              weight: MAP.bund.weight,
              fillColor: MAP.bund.color,
              fillOpacity: MAP.bund.fillOpacity,
            }}
          />
        )}

        {/* Where the excavated soil goes */}
        {earthwork?.haul_plan?.arrows?.map((arrow, index) => (
          <Polyline
            key={`haul-${index}`}
            positions={[
              [arrow.from[1], arrow.from[0]],
              [arrow.to[1], arrow.to[0]],
            ]}
            pathOptions={{ color: MAP.haul.color, weight: MAP.haul.weight, opacity: 0.85 }}
          >
            <Popup>
              {arrow.volume_m3.toLocaleString('en-IN')} m³ moved {arrow.distance_m} m
            </Popup>
          </Polyline>
        ))}

        {/* Numbered pins, always on top */}
        {layers.sites &&
          sites.map((site) => (
            <Marker
              key={`pin-${site.site_id}-${site.site_id === selectedSiteId}`}
              position={[site.location.lat, site.location.lon]}
              icon={siteIcon(site.rank, site.site_id === selectedSiteId)}
              zIndexOffset={site.site_id === selectedSiteId ? 1000 : 0}
              eventHandlers={{ click: () => onSelectSite(site.site_id) }}
            >
              <Popup>
                <div className="popup-content">
                  <h3>
                    #{site.rank} · {site.structure?.label ?? site.kind}
                  </h3>
                  <div className="popup-stat">
                    <span className="label">Holds</span>
                    <span className="value">
                      {Math.round(site.storage.capacity_m3).toLocaleString('en-IN')} m³
                    </span>
                  </div>
                  <div className="popup-stat">
                    <span className="label">Catchment</span>
                    <span className="value">{site.catchment.area_ha.toFixed(1)} ha</span>
                  </div>
                  <div className="popup-stat">
                    <span className="label">Collects a year</span>
                    <span className="value">
                      {Math.round(site.yield.harvestable_mean_m3).toLocaleString('en-IN')} m³
                    </span>
                  </div>
                  <div className="popup-stat">
                    <span className="label">Confidence</span>
                    <span className="value">{site.confidence_label}</span>
                  </div>
                </div>
              </Popup>
            </Marker>
          ))}
      </MapContainer>

      <MapControls
        analysis={analysis}
        earthwork={earthwork}
        layers={layers}
        onToggleLayer={onToggleLayer}
        selected={selected}
        zoom={zoom}
      />
    </>
  );
}

/**
 * Zoom buttons, the layer switches and the legend in one column.
 *
 * The switches and the legend are the same control: each row carries the
 * swatch it turns on, so a reader never has to match a separate legend entry
 * against a separate checkbox.
 */
function MapControls({ analysis, earthwork, layers, onToggleLayer, selected, zoom }) {
  const items = [
    { key: 'sites', label: 'The chosen pond', swatch: 'pond' },
    { key: 'catchment', label: 'Catchment of your area', swatch: 'catchment' },
    { key: 'siteCatchment', label: 'Catchment of chosen pond', swatch: 'site-catchment' },
    { key: 'focus', label: 'Dim outside my area', swatch: 'focus' },
    { key: 'streams', label: 'Drainage lines', swatch: 'stream' },
    { key: 'constraints', label: 'Cannot build here', swatch: 'constraint' },
    { key: 'village', label: 'Village extent', swatch: 'village' },
  ];

  if (!analysis) return null;

  return (
    <div className="map-panel">
      <div className="map-panel-head">
        <span>Map layers</span>
        <span className="zoom-level" title="Zoom level">z{zoom}</span>
      </div>
      {items.map((item) => (
        <label key={item.key} className="map-layer-row">
          <input
            type="checkbox"
            checked={!!layers[item.key]}
            onChange={() => onToggleLayer(item.key)}
          />
          <span className={`swatch swatch-${item.swatch}`} aria-hidden="true" />
          <span className="map-layer-label">{item.label}</span>
        </label>
      ))}

      <div className="map-panel-notes">
        {selected && (
          <div className="map-note">
            <span className="swatch swatch-selected" aria-hidden="true" />
            Showing pond {selected.rank}. Pick another from the list, or tap a pin.
          </div>
        )}
        {earthwork?.new_footprint && (
          <div className="map-note">
            <span className="swatch swatch-after" aria-hidden="true" />
            Dashed amber: the pond after the works
          </div>
        )}
        {analysis?.catchment?.truncated && (
          <div className="map-note map-note-warn">
            The catchment is dashed where it leaves the analysed area
          </div>
        )}
      </div>
    </div>
  );
}
