// Groups records by location_name. Records are already sorted newest-first
// by the backend, so the first entry in each group is that location's most
// recent record.
export function groupByLocation(records) {
  const map = new Map();
  records.forEach(r => {
    if (!map.has(r.location_name)) map.set(r.location_name, []);
    map.get(r.location_name).push(r);
  });
  return map;
}
