// Thin wrapper around a Material Symbols Outlined glyph, matching the icon
// set the verified Stitch screens were authored with.
export default function Icon({ name, fill, className = '', style }) {
  return (
    <span className={`material-symbols-outlined${fill ? ' icon-fill-1' : ''} ${className}`} style={style}>
      {name}
    </span>
  );
}
