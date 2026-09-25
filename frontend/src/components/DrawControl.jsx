import { useEffect, useRef } from 'react';
import { useMap } from 'react-leaflet';
import * as turf from '@turf/turf';
import '@geoman-io/leaflet-geoman-free';
import '@geoman-io/leaflet-geoman-free/dist/leaflet-geoman.css';

const MIN_AREA_HA = 1;
const MAX_AREA_KM2 = 25;

/**
 * Polygon and rectangle drawing, backed by Leaflet-Geoman.
 *
 * Geoman is wired imperatively through `map.pm` rather than through a wrapper
 * component: react-leaflet v5 exposes the map instance directly, and this keeps
 * one less dependency in the tree.
 *
 * Only one selection exists at a time, and the area is checked here before the
 * request goes out, so an oversized or self-intersecting shape is rejected with
 * an explanation instead of a server error.
 */
export default function DrawControl({ onChange, onInvalid, externalGeometry }) {
  const map = useMap();
  const layerRef = useRef(null);
  // The Geoman handlers are registered once, so the current callbacks are held
  // in refs.  Writing them from an effect rather than during render keeps the
  // component safe under concurrent rendering.
  const onChangeRef = useRef(onChange);
  const onInvalidRef = useRef(onInvalid);

  useEffect(() => {
    onChangeRef.current = onChange;
    onInvalidRef.current = onInvalid;
  }, [onChange, onInvalid]);

  useEffect(() => {
    if (!map?.pm) return undefined;

    map.pm.addControls({
      position: 'topleft',
      drawMarker: false,
      drawCircle: false,
      drawCircleMarker: false,
      drawPolyline: false,
      drawText: false,
      cutPolygon: false,
      rotateMode: false,
      drawPolygon: true,
      drawRectangle: true,
      editMode: true,
      dragMode: true,
      removalMode: true,
    });

    map.pm.setGlobalOptions({
      snappable: true,
      snapDistance: 20,
      allowSelfIntersection: false,
      finishOn: 'dblclick',
      templineStyle: { color: '#ffffff', weight: 2 },
      hintlineStyle: { color: '#ffffff', weight: 2, dashArray: '5,5' },
      // White, because the area the user drew belongs to no result layer and
      // should never be mistaken for one.
      pathOptions: {
        color: '#ffffff',
        weight: 2.5,
        fillColor: '#ffffff',
        fillOpacity: 0.06,
      },
    });

    const validate = (layer) => {
      let geojson;
      try {
        geojson = layer.toGeoJSON();
      } catch {
        return null;
      }
      const geometry = geojson.geometry;
      if (!geometry || geometry.type !== 'Polygon') return null;

      if (turf.kinks(geojson).features.length > 0) {
        onInvalidRef.current?.('That shape crosses itself. Redraw it without crossing edges.');
        return null;
      }

      const areaM2 = turf.area(geojson);
      if (areaM2 < MIN_AREA_HA * 10000) {
        onInvalidRef.current?.(
          `That area is ${(areaM2 / 10000).toFixed(2)} ha; the minimum is ${MIN_AREA_HA} ha.`,
        );
        return null;
      }
      if (areaM2 > MAX_AREA_KM2 * 1e6) {
        onInvalidRef.current?.(
          `That area is ${(areaM2 / 1e6).toFixed(1)} km²; the maximum is ${MAX_AREA_KM2} km². Draw a smaller area.`,
        );
        return null;
      }
      return { geometry, areaM2 };
    };

    const replaceSelection = (layer) => {
      if (layerRef.current && layerRef.current !== layer) {
        try {
          map.removeLayer(layerRef.current);
        } catch {
          /* already gone */
        }
      }
      layerRef.current = layer;
    };

    const handleCreate = (event) => {
      const { layer } = event;
      const result = validate(layer);
      if (!result) {
        try {
          map.removeLayer(layer);
        } catch {
          /* nothing to remove */
        }
        return;
      }
      replaceSelection(layer);
      onChangeRef.current?.(result.geometry, result.areaM2);

      layer.on('pm:edit', () => {
        const edited = validate(layer);
        if (edited) onChangeRef.current?.(edited.geometry, edited.areaM2);
      });
      layer.on('pm:dragend', () => {
        const dragged = validate(layer);
        if (dragged) onChangeRef.current?.(dragged.geometry, dragged.areaM2);
      });
    };

    const handleRemove = (event) => {
      if (event.layer === layerRef.current) {
        layerRef.current = null;
        onChangeRef.current?.(null, 0);
      }
    };

    map.on('pm:create', handleCreate);
    map.on('pm:remove', handleRemove);

    return () => {
      map.off('pm:create', handleCreate);
      map.off('pm:remove', handleRemove);
      try {
        map.pm.removeControls();
      } catch {
        /* map already torn down */
      }
    };
  }, [map]);

  // A boundary chosen from search can seed the selection, so the user can
  // analyse a village straight away and adjust the shape afterwards.
  useEffect(() => {
    if (!map?.pm || !externalGeometry) return;
    const L = window.L;
    if (!L) return;

    if (layerRef.current) {
      try {
        map.removeLayer(layerRef.current);
      } catch {
        /* already gone */
      }
      layerRef.current = null;
    }

    const layer = L.geoJSON(
      { type: 'Feature', geometry: externalGeometry, properties: {} },
      {
        style: {
          color: '#ffffff',
          weight: 2,
          fillColor: '#ffffff',
          fillOpacity: 0.05,
          dashArray: '6,4',
        },
      },
    );
    layer.addTo(map);
    layer.eachLayer?.((child) => {
      child.pm?.enable?.({ allowSelfIntersection: false });
      child.pm?.disable?.();
    });
    layerRef.current = layer;
  }, [map, externalGeometry]);

  return null;
}
