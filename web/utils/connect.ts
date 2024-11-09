import { PrismaClient } from "@prisma/client";

const PrismaClientInstance = () => new PrismaClient();

declare global {
  var prisma: ReturnType<typeof PrismaClientInstance>;
  var prismaGlobal: ReturnType<typeof PrismaClientInstance> | undefined;
}

const prisma = globalThis.prismaGlobal || PrismaClientInstance();

export default prisma;

if (process.env.NODE_ENV !== "production") {
  globalThis.prismaGlobal = prisma;
}
