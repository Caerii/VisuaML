/** @fileoverview Defines the RemoteCursor component, which displays a visual cursor and name label for other users interacting with the canvas in a collaborative session. The cursor position and user details are provided as props. */
import React from 'react';
import styles from './styles/RemoteCursor.module.css';

interface RemoteCursorProps {
  x: number;
  y: number;
  name: string;
  color: string;
}

export const RemoteCursor: React.FC<RemoteCursorProps> = ({ x, y, name, color }) => {
  return (
    <div
      className={styles.remoteCursorContainer}
      style={{
        left: `${x}px`,
        top: `${y}px`,
      }}
    >
      {/* Cursor pointer SVG */}
      <svg
        className={styles.cursorSvg}
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none" // Fill is part of path
      >
        <path
          d="M3 3L10.07 19.97L12.58 12.58L19.97 10.07L3 3Z"
          fill={color} // Dynamic fill color for the cursor itself
          stroke="white"
          strokeWidth="1"
          strokeLinejoin="round"
        />
      </svg>

      {/* User name label */}
      <div
        className={styles.nameLabel}
        style={{
          backgroundColor: color, // Dynamic background color for the label
        }}
      >
        {name}
      </div>
    </div>
  );
};
