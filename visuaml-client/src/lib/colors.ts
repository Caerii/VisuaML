/**
 * Generates a deterministic HSL color string from an input string (e.g., user ID).
 * @param str The input string.
 * @returns An HSL color string (e.g., "hsl(120, 70%, 80%)").
 */
export const stringToHSL = (
  str: string,
  saturation: number = 70, // Range: 0-100
  lightness: number = 75, // Range: 0-100
): string => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    // Simple hash: multiply by a prime, add char code, keep it integer
    hash = (hash * 31 + str.charCodeAt(i)) | 0;
  }
  const hue = Math.abs(hash % 360); // Hue: 0-359
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
};
