import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "@/pages/Home";
import SteadyState from "@/pages/SteadyState";
import QuasiSteady from "@/pages/QuasiSteady";
import Cooling from "@/pages/Cooling";
import Admin from "@/pages/Admin";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/steady" element={<SteadyState />} />
        <Route path="/quasi" element={<QuasiSteady />} />
        <Route path="/cooling" element={<Cooling />} />
        <Route path="/admin" element={<Admin />} />
      </Routes>
    </Router>
  );
}
