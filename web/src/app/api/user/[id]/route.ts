import { NextRequest, NextResponse } from "next/server";

const axios = require("axios");

export const POST = async (req: NextRequest) => {
  try {
    const id = await req;
    return NextResponse.json({ hello: id }, { status: 200 });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error }, { status: 500 });
  }
};
