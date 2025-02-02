import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import axios from "axios";
import TextProcessor from "../components/TextProcessor";

// Mock axios globally
jest.mock("axios");
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Suppress console.error during tests
beforeAll(() => {
  jest.spyOn(console, "error").mockImplementation(() => {});
});

afterAll(() => {
  jest.restoreAllMocks();
});

// Render the TextProcessor component wrapped with MemoryRouter
const renderComponent = () => {
  render(
    <MemoryRouter>
      <TextProcessor />
    </MemoryRouter>
  );
};

describe("TextProcessor Integration Tests", () => {
  test("renders TextProcessor with input area and buttons", () => {
    renderComponent();

    expect(
      screen.getByPlaceholderText("Enter your text here...")
    ).toBeInTheDocument();
    expect(screen.getByText("Smart AI Explanation")).toBeInTheDocument();
    expect(screen.getByText("Detailed Insights Report")).toBeInTheDocument();
    expect(screen.getByText("Upload File")).toBeInTheDocument();
  });

  test("handles API error gracefully", async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error("Network Error"));

    renderComponent();
    fireEvent.change(screen.getByPlaceholderText("Enter your text here..."), {
      target: { value: "Test text" },
    });
    fireEvent.click(screen.getByText("Detailed Insights Report"));

    await waitFor(() =>
      expect(
        screen.getByText(
          "An error occurred while generating the SHAP + LIME explanation."
        )
      ).toBeInTheDocument()
    );
  });

  test("handles successful API response", async () => {
    mockedAxios.post.mockResolvedValueOnce({
      status: 200,
      data: {
        prediction: "AI-generated",
        explanation: "This text seems AI-generated due to...",
      },
    });

    renderComponent();
    fireEvent.change(screen.getByPlaceholderText("Enter your text here..."), {
      target: { value: "Sample text for classification." },
    });
    fireEvent.click(screen.getByText("Smart AI Explanation"));

    await waitFor(() => {
      expect(mockedAxios.post).toHaveBeenCalledWith(
        "http://localhost:5000/api/classify",
        { text: "Sample text for classification." }
      );
    });
  });

  test("shows error when trying to classify without input", () => {
    renderComponent();
    fireEvent.click(screen.getByText("Smart AI Explanation"));

    expect(
      screen.getByText("Please input text or upload a valid file.")
    ).toBeInTheDocument();
  });
});
