module.exports = {
  extends: [
    'airbnb-typescript/base',
    'plugin:react/recommended',
    'plugin:tailwindcss/recommended',
    'prettier',
  ],
  parserOptions: { project: './tsconfig.app.json' },
  rules: { 'react/react-in-jsx-scope': 'off' },
};
