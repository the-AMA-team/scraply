import { NextRequest, NextResponse } from "next/server";
import prisma from '../../../../util/connect'

export const GET = async (req: NextRequest, { params }: any) => {
  try {
    const { teacherEmail } = await params;

    const classrooms = await prisma.classrooms.findMany({
      where: {
        teacher: teacherEmail,
      },
    });

    // console.log("Classrooms found:", classrooms);

    // if (!user) {
    //   console.log("User not found");
    //   return null;
    // }

    // console.log("User found:", user);
    return NextResponse.json({ classrooms: classrooms }, { status: 200 });
  } catch (error) {
    console.error("Error retrieving user:", error);
    return NextResponse.json({ error }, { status: 200 });

    throw error;
  }
};
