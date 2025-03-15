import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import ExplanationReport from "../components/ExplanationReport";

describe("ExplanationReport Tests", () => {
  test("Renders HTML content in an iframe", () => {
    const mockHtml = "<h1>Report</h1>";

    render(
      <MemoryRouter
        initialEntries={[
          { pathname: "/explanation-report", state: { htmlContent: mockHtml } },
        ]}
      >
        <Routes>
          <Route path="/explanation-report" element={<ExplanationReport />} />
        </Routes>
      </MemoryRouter>
    );

    const iframe = screen.getByTestId(
      "explanation-iframe"
    ) as HTMLIFrameElement;
    expect(iframe).toBeInTheDocument();
    expect(iframe).toHaveAttribute("srcdoc", mockHtml);
  });

  test("Handles missing htmlContent gracefully", () => {
    render(
      <MemoryRouter initialEntries={[{ pathname: "/explanation-report" }]}>
        <Routes>
          <Route path="/explanation-report" element={<ExplanationReport />} />
        </Routes>
      </MemoryRouter>
    );

    const iframe = screen.getByTestId(
      "explanation-iframe"
    ) as HTMLIFrameElement;
    expect(iframe).toBeInTheDocument();

    const srcDocVal = iframe.getAttribute("srcdoc");
    expect(srcDocVal === null || srcDocVal === "").toBe(true); // Corrected assertion
  });
});

