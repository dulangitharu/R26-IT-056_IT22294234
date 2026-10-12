export default function LoadingState({ text = 'Loading...' }) {
  return (
    <div className="no-det">
      <span className="spinner" />{text}
    </div>
  );
}
