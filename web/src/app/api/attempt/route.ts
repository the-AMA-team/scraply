import { NextRequest, NextResponse } from "next/server";
import prisma from "../../../../utils/connect";

const axios = require("axios");

// code for updating the attempts
export const POST = async (req: NextRequest) => {
  try {
    const { userId, attemptId, updatedData  } = await req.json();

    // const updatedAttempt = await prisma.attempt.updateMany({
    //     where: {
    //         userId: "004",
    //         id: "672c3d5ac11adb72d492e34c"
    //     },
    //     data:  {"level":"L1","rating":-1,"archetecture":"","lastLoss":-1}
    // });

    const updatedAttempt = await prisma.attempt.updateMany({
        where: {
            userId: userId,
            id: attemptId
        },
        data: updatedData
    });

    if (updatedAttempt.count == 0) {
        return NextResponse.json({ error : "Not Found"} , {status: 404});
    }
    return NextResponse.json({ updatedAttempt: updatedAttempt }, { status: 200 });
  } catch (error) {
    // console.error(error);
    return NextResponse.json({ error }, { status: 500 });
  }
};