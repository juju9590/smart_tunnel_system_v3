import React, { useMemo } from "react";
import TunnelModule from "./modules/tunnel";

function App() {
  const host = useMemo(() => window.location.hostname || "localhost", []);
  return <TunnelModule host={host} />;
}

export default App;