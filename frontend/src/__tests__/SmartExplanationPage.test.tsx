import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import SmartExplanationPage from "../pages/SmartExplanationPage";

describe("SmartExplanationPage Tests", () => {
  test("Displays predicted label and explanation from route state", () => {
    render(
      <MemoryRouter
        initialEntries={[
          {
            pathname: "/smart-explanation",
            state: {
              classificationResult: {
                prediction: "Positive",
                explanation: "It is positive because...",
              },
            },
          },
        ]}
      >
        <SmartExplanationPage />
      </MemoryRouter>
    );

    // Check for the label and its value separately
    expect(screen.getByText("Predicted Label:")).toBeInTheDocument();
    expect(screen.getByText("Positive")).toBeInTheDocument();
    expect(screen.getByText("It is positive because...")).toBeInTheDocument();
  });

  test('Displays "N/A" and fallback when no state is passed', () => {
    render(
      <MemoryRouter>
        <SmartExplanationPage />
      </MemoryRouter>
    );

    // Check for fallback values
    expect(screen.getByText("Predicted Label:")).toBeInTheDocument();
    expect(screen.getByText("N/A")).toBeInTheDocument();
    expect(screen.getByText("Explanation #1 missing")).toBeInTheDocument();
  });
});

