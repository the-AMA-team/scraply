import { systemPrompt, userPrompt } from "./prompts";
import OpenAI from "openai";
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  try {
    const { dataset } = await req.json();

    const completion = await openai.beta.chat.completions.parse({
      model: "gpt-4o-2024-08-06",
      messages: [
        { role: "system", content: systemPrompt() },
        { role: "user", content: userPrompt(dataset) },
      ],
      temperature: 0.2,
    });

    const response = completion.choices[0]?.message.content;
    console.log(response);

    if (response) {
      const data = JSON.parse(response);

      return NextResponse.json(data);
    }

    return NextResponse.json({ error: "no response" }, { status: 404 });
  } catch (error) {
    console.error("Error: ", error);
    return NextResponse.json({ error: "bad request" }, { status: 500 });
  }
}
