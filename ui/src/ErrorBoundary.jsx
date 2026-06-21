import { Component } from 'react';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          gap: '12px',
          color: '#e74c3c',
          fontFamily: 'monospace',
          padding: '40px',
          textAlign: 'center',
        }}>
          <div style={{ fontSize: '32px' }}>&#9888;</div>
          <div style={{ fontSize: '16px', fontWeight: 700 }}>Something went wrong</div>
          <div style={{ fontSize: '13px', color: '#aaa', maxWidth: '500px' }}>
            {String(this.state.error?.message ?? this.state.error)}
          </div>
          <button
            onClick={() => this.setState({ error: null })}
            style={{
              marginTop: '8px',
              padding: '6px 18px',
              background: 'rgba(231,76,60,0.15)',
              border: '1px solid #e74c3c',
              borderRadius: '4px',
              color: '#e74c3c',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
